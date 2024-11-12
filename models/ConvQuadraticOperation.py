import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvQuadraticOperation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 bias: bool = True):
        super(ConvQuadraticOperation, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_r = nn.Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))
        self.weight_g = nn.Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))
        self.weight_b = nn.Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))

        if bias:
            self.bias_r = nn.Parameter(torch.empty(out_channels))
            self.bias_g = nn.Parameter(torch.empty(out_channels))
            self.bias_b = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_g', None)
            self.register_parameter('bias_b', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        if self.bias_r is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_r, -bound, bound)
            nn.init.constant_(self.bias_g, 1)
            nn.init.constant_(self.bias_b, 0)

    def forward(self, x):
        out = F.conv1d(x, self.weight_r, self.bias_r, self.stride, self.padding, 1, 1) \
              * F.conv1d(x, self.weight_g, self.bias_g, self.stride, self.padding, 1, 1) \
              + F.conv1d(torch.pow(x, 2), self.weight_b, self.bias_b, self.stride, self.padding, 1, 1)
        return out
