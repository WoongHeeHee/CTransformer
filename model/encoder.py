import torch
import torch.nn as nn
from copy import deepcopy
from .attention import MultiHeadAttentionLayer
from .feedforward import PositionWiseFeedForwardLayer


class ResidualConnectionLayer(nn.Module):
    """
    Residual connection layer with layer normalization
    """
    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out + x
        return x


class EncoderBlock(nn.Module):
    """
    Single encoder block with self-attention and feed-forward layers
    """
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = nn.ModuleList([ResidualConnectionLayer() for _ in range(2)])

    def forward(self, src, src_mask):
        out = src
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residuals[1](out, self.position_ff)
        return out


class Encoder(nn.Module):
    """
    Complete encoder with multiple encoder blocks
    """
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_block) for _ in range(n_layer)])

    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out