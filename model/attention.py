import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math
from .complex_linear import ComplexLinear


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-Head Attention Layer with Complex Linear layers
    - Query, Key, Value projection
    - Scaled dot-product attention
    - Linear projection after concatenation 
    """
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = deepcopy(qkv_fc)
        self.k_fc = deepcopy(qkv_fc)
        self.v_fc = deepcopy(qkv_fc)
        self.out_fc = out_fc

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
        n_batch = query.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model // self.h)
            out = out.transpose(1, 2)
            return out

        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        out = self.calculate_attention(query, key, value, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.out_fc(out)
        return out

    def calculate_attention(self, query, key, value, mask):
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1))
        attention_score = attention_score / math.sqrt(d_k)

        # attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_prob = F.softmax(abs(attention_score), dim=-1)
        out = torch.matmul(attention_prob, value.real) + 1j * torch.matmul(attention_prob, value.imag)
        # attention_prob = F.softmax(attention_score, dim=-1)
        # out = torch.matmul(attention_prob, value)
        return out