import torch
import torch.nn as nn

class PostionWiseFeedForwardLayer(nn.Module):
    """
    Feed-forward layer applied to each postition separately and identically
    Usually: Linear -> ReLU -> Linear
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # TODO: define two linear layers
        self.d_model = d_model
        self.d_ff = d_ff
        pass

    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            output: same shape as x
        """
        # TODO: implement forward pass with activation
        pass