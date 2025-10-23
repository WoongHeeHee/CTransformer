import torch
import torch.nn as nn
from .complex_linear import ComplexLinear


class PositionWiseFeedForwardLayer(nn.Module):
    """
    Feed-forward layer applied to each position separately and identically
    Uses Complex Linear layers with cardioid activation
    """
    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1 # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2 # (d_ff, d_embed)

    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            output: same shape as x
        """
        out = x
        out = self.fc1(out)
        out = self.cardioid(out)
        out = self.fc2(out)
        return out

    def zrelu(self, z): #z : tensor 활성화 함수 1
        real_parts = z.real
        imag_parts = z.imag

        mask = (real_parts > 0) & (imag_parts > 0)

        filtered_tensor = torch.where(mask, z, torch.zeros_like(z))

        return filtered_tensor
    
    def cardioid(self, z):
        real_parts = z.real
        imag_parts = z.imag

        theta = torch.atan2(imag_parts, real_parts)

        return (1 + torch.cos(theta)) * z / 2