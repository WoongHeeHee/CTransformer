import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
import math


class ComplexLinear(nn.Module):
    """
    Complex linear layer for complex field calculations.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    real_weight: Tensor
    imag_weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.real_weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.imag_weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.real_bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.imag_bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('real_bias', None)
            self.register_parameter('imag_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.real_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.imag_weight, a=math.sqrt(5))
        if self.real_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.real_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.real_bias, -bound, bound)
            nn.init.uniform_(self.imag_bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        input_real = input.real
        input_imag = input.imag

        real_output = torch.matmul(input_real, self.real_weight.T) - torch.matmul(input_imag, self.imag_weight.T)
        imag_output = torch.matmul(input_real, self.imag_weight.T) + torch.matmul(input_imag, self.real_weight.T)

        if self.real_bias is not None:
            real_output += self.real_bias
            imag_output += self.imag_bias

        return torch.complex(real_output, imag_output)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.real_bias is not None}' 