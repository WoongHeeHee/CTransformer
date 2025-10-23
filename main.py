# main.py

import torch
from config import d_embed, d_model, h, d_ff, n_layer, batch_size, num_epochs, train_data_path
from data import preprocess_data
from train import build_model, Trainer, get_device


def main():
    """
    Main function to run the CTransformer training
    """
    print("=== CTransformer Training ===")
    print(f"Using device: {get_device()}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    dataloader, word2idx, embedding_layer = preprocess_data(
        path=train_data_path, 
        d_embed=d_embed, 
        batch_size=batch_size
    )
    print(f"Vocabulary size: {len(word2idx)}")
    
    # Build model
    print("Building model...")
    model = build_model(d_embed, d_model, h, d_ff, n_layer)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = Trainer(model, device=get_device())
    
    # Train model
    print("Starting training...")
    trainer.train(dataloader, num_epochs=num_epochs, learning_rate=0.01)
    
    # Evaluate model
    print("Evaluating model...")
    trainer.evaluate(dataloader)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
