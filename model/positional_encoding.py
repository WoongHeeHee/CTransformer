import torch
import torch.nn

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # TODO: precompute positional encodings and register as buffer
        pass

    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            x + positional_encoding
        """
        # TODO: add positional encodings to input
        pass