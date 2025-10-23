import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Implements complex sinusoidal positional encoding
    """
    def __init__(self, d_embed=512, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding_requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(d_embed) * -(math.log(10000.0) / d_embed))
        encoding[:, :] = torch.exp(1j * position * div_term) # Switch to Complex Field
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            x + positional_encoding
        """
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out