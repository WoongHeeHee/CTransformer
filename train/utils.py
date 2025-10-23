import torch
import torch.nn as nn
from model import (
    Encoder, EncoderBlock, MultiHeadAttentionLayer, 
    PositionWiseFeedForwardLayer, ComplexLinear
)


def build_model(d_embed=512, d_model=512, h=8, d_ff=2048, n_layer=6):
    """
    Build the complete CTransformer model
    """
    # Create attention layer
    attention = MultiHeadAttentionLayer(
        d_model=d_model, 
        h=h, 
        qkv_fc=ComplexLinear(d_embed, d_model), 
        out_fc=ComplexLinear(d_model, d_embed)
    )
    
    # Create feed-forward layer
    position_ff = PositionWiseFeedForwardLayer(
        fc1=ComplexLinear(d_embed, d_ff), 
        fc2=ComplexLinear(d_ff, d_embed)
    )
    
    # Create encoder block
    encoder_block = EncoderBlock(
        self_attention=attention,
        position_ff=position_ff
    )
    
    # Create encoder
    model = Encoder(encoder_block, n_layer)
    
    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """
    Save model to file
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """
    Load model from file
    """
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model


def get_device():
    """
    Get the best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
