import torch
import torch.nn as nn

class ComplexLinear(nn.Module):
    """
    Casual linear layer calculate in Complex Field.
    """
    def __init__(self):
        super().__init__()
        # TODO: 