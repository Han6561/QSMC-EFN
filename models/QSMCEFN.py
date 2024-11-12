from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.dirichlet import Dirichlet
from torch.nn import functional as F
from models.ConvQuadraticOperation import ConvQuadraticOperation
# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1,
#                  padding=None, use_norm=True, use_act=True):
#         super().__init__()
#         block = []
#         padding = padding or kernel_size // 2
#         block.append(nn.Conv1d(
#             in_channel, out_channel, kernel_size, stride, padding=padding, groups=groups, bias=False
#         ))
#         if use_norm:
#             block.append(nn.BatchNorm1d(out_channel))
#         if use_act:
#             block.append(nn.GELU())
#
#         self.block = nn.Sequential(*block)
#
#     def forward(self, x):
#         return self.block(x)
class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=None, use_norm=True, use_act=True):
        super(ConvBNReLU, self).__init__()
        block = []
        padding = padding or kernel_size // 2
        block.append(ConvQuadraticOperation(
            in_channel, out_channel, kernel_size, stride, padding, bias=False
        ))
        if use_norm:
            block.append(nn.BatchNorm1d(out_channel))
        if use_act:
            block.append(nn.GELU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.layernorm(x)
        return x.transpose(-1, -2)

class Add(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(Add, self).__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.ReLU()

    def forward(self, x):
        w = self.w_relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return weight[0] * x[0] + weight[1] * x[1]

class Embedding(nn.Module):
    def __init__(self, d_in, d_out, stride=2, n=4):
        super(Embedding, self).__init__()
        d_hidden = d_out // n
        self.conv1 = nn.Conv1d(d_in, d_hidden, 1, 1)
        self.sconv = nn.ModuleList([
            nn.Conv1d(d_hidden, d_hidden, 2 * i + 2 * stride - 1,
                      stride=stride, padding=stride + i - 1, groups=d_hidden, bias=False)
            for i in range(n)])
        self.act_bn = nn.Sequential(
            nn.BatchNorm1d(d_out), nn.GELU())

    def forward(self, x):
        signals = []
        x = self.conv1(x)
        for sconv in self.sconv:
            signals.append(sconv(x))
        x = torch.cat(signals, dim=1)
        return self.act_bn(x)

class BroadcastAttention(nn.Module):
    def __init__(self,
                 dim,
                 proj_drop=0.,
                 attn_drop=0.,
                 qkv_bias=True):
        super().__init__()
        self.dim = dim

        self.qkv_proj = nn.Conv1d(dim, 1 + 2 * dim, kernel_size=1, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.dim, self.dim], dim=1
        )

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(value) * context_vector.expand_as(value)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class BA_FFN_Block(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 drop=0.,
                 attn_drop=0.):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.add1 = Add()
        self.attn = BroadcastAttention(dim=dim,
                                       attn_drop=attn_drop,
                                       proj_drop=drop)

        self.norm2 = LayerNorm(dim)
        self.add2 = Add()
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, ffn_dim, 1, 1, bias=True),
            nn.GELU(),
            nn.Dropout(p=drop),
            nn.Conv1d(ffn_dim, dim, 1, 1, bias=True),
            nn.Dropout(p=drop)
        )

    def forward(self, x):
        x = self.add1([self.attn(self.norm1(x)), x])
        x = self.add2([self.ffn(self.norm2(x)), x])
        return x

class LFEL(nn.Module):
    def __init__(self, d_in, d_out, drop):
        super(LFEL, self).__init__()

        self.embed = Embedding(d_in, d_out, stride=2, n=4)
        self.block = BA_FFN_Block(dim=d_out,
                                  ffn_dim=d_out // 4,
                                  drop=drop,
                                  attn_drop=drop)

    def forward(self, x):
        x = self.embed(x)
        return self.block(x)

class SeparableMultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(SeparableMultiScaleConv, self).__init__()
        self.depthwise_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=ks, padding=ks//2, groups=in_channels)
            for ks in kernel_sizes
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            for _ in kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for dw_conv, pw_conv in zip(self.depthwise_convs, self.pointwise_convs):
            out = dw_conv(x)
            out = pw_conv(out)
            outputs.append(out)
        return outputs  # Return features from different scales

# class DirichletLayer(nn.Module):
#     def __init__(self, alpha_init):
#         super(DirichletLayer, self).__init__()
#         self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
#
#     def forward(self, x):
#         dirichlet = Dirichlet(self.alpha + x)
#         return dirichlet.sample()

# class DS_EvidenceFusion(nn.Module):
#     def __init__(self):
#         super(DS_EvidenceFusion, self).__init__()
#
#     def forward(self, evidence1, evidence2):
#         combined_evidence = evidence1 * evidence2
#         normalization = torch.sum(combined_evidence, dim=1, keepdim=True)
#         return combined_evidence / normalization
import torch
import torch.nn as nn


class DS_EvidenceFusion(nn.Module):
    def __init__(self):
        super(DS_EvidenceFusion, self).__init__()

    def forward(self, evidence1, evidence2):
        # Step 1: Calculate combined evidence
        combined_evidence = evidence1 * evidence2

        # Step 2: Calculate the conflict value
        conflict = torch.sum(evidence1 * (1 - evidence2) + evidence2 * (1 - evidence1), dim=1, keepdim=True)

        # Step 3: Normalize with conflict handling (1 - conflict)
        normalization = torch.sum(combined_evidence, dim=1, keepdim=True)
        normalized_evidence = combined_evidence / (normalization + conflict + 1e-8)  # avoid division by zero

        return normalized_evidence


class QSMCEFN(nn.Module):
    def __init__(self, _, in_channel, out_channel, drop=0.1, dim=32):
        super(QSMCEFN, self).__init__()

        self.in_layer = nn.Sequential(
            nn.AvgPool1d(2, 2),
            ConvBNReLU(in_channel, dim, kernel_size=15, stride=2)
        )

        self.multi_scale_conv = SeparableMultiScaleConv(dim, dim, kernel_sizes=[3, 5 , 7])

        self.LFELs = nn.Sequential(
            LFEL(dim, 2 * dim, drop),
            LFEL(2 * dim, 4 * dim, drop),
            LFEL(4 * dim, 8 * dim, drop),
            nn.AdaptiveAvgPool1d(1)
        )


        self.out_layer = nn.Linear(8 * dim, out_channel)

        # self.dirichlet_layer = DirichletLayer(alpha_init=[1.0] * out_channel)
        self.evidence_fusion = DS_EvidenceFusion()

    def forward(self, x):
        x = self.in_layer(x)
        multi_scale_features = self.multi_scale_conv(x)

        fused_features = multi_scale_features[0]
        for feature in multi_scale_features[1:]:
            evidence1 = F.softmax(fused_features, dim=-1)
            evidence2 = F.softmax(feature, dim=-1)
            fused_features = self.evidence_fusion(evidence1, evidence2)

        fused_features = self.LFELs(fused_features)
        x = self.out_layer(fused_features.squeeze())
        return x






