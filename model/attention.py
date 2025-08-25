import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-Head Attention Layer
    - Query, Key, Value projection
    - Scaled dot-product attention
    - Linear projection after concatenation 
    """
    def __init__(self, d_model: int, h: int):
        super().__init__()
        # TODO: define weight matrices for Q, K, V and final linear layer
        pass

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: shape (batch, seq_len, d_model)
            key:   shape (batch, seq_len, d_model)
            value: shape (batch, seq_len, d_model)
            mask: optional attention mask
        Returns:
            output: attended values
        """
        # TODO: implement scaled dot-product attention
        pass