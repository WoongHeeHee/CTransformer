# main.py

# Model import #
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from torchtext.data import get_tokenizer

# Hyperparameter Loading #
from config import d_embed, d_model, h, d_ff, n_layer, batch_size, num_epochs
from model import Encoder, EncoderBlock, MultiHeadAttentionLayer, PositionWiseFeedForwardLayer, PositionalEncoding, ComplexLinear


# Preprocessing function #
def load_and_preprocess_data(path='train.tsv'):
    df = pd.read_csv(path, sep='\t')
    df.columns = ['Dummy', 'Label', 'Dummy2', 'Sentence'] # Label -> 1: acceptable
    df = df.drop(columns=['Dummy', 'Dummy2'])

    tokenizer = get_tokenizer('basic_english')
    df['Tokenized'] = df['Sentence'].apply(tokenizer)

    vocab = set(word for sentence in df['Tokenized'] for word in sentence)
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    embedding = nn.Embedding(len(vocab), d_embed)

    def embed_sentence(sentence):
        indices = torch.tensor([word2idx[word] for word in sentence], dtype=torch.long)
        return embedding(indices).detach()

    df['embedded'] = df['Tokenized'].apply(embed_sentence) # one sentence have one embedded vector(tensor)
    padded = pad_sequence(df['embedded'].tolist(), batch_first=True)

    x = padded.to(torch.cfloat)
    x = PositionalEncoding(d_embed=d_embed, max_len=padded.size(1))(x)
    y = torch.tensor(df['Label'].values, dtype=torch.float32)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Model Building #
def build_model():
    attention = MultiHeadAttentionLayer(
        d_model=d_model, h=h,
        qkv_fc=ComplexLinear(d_embed, d_model),
        out_fc=ComplexLinear(d_model, d_embed)
    )

    ff_layer = PositionWiseFeedForwardLayer(
        fc1=ComplexLinear(d_embed, d_ff),
        fc2=ComplexLinear(d_ff, d_embed)
    )

    encoder_block = EncoderBlock(self_attention=attention, position_ff=ff_layer)
    model = Encoder(encoder_block, n_layer=n_layer)
    return model


# Training Loop #
def train(model, dataloader, num_epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            src_mask = None
            output = model(batch_x, src_mask)
            output = output.mean(dim=1)
            output = ComplexLinear(d_model, 1)(output)
            output = torch.sigmoid(abs(output)).squeeze(-1)

            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")


# === main 진입점 === #
def main():
    print("Loading data...")
    dataloader = load_and_preprocess_data()

    print("Building model...")
    model = build_model()

    print("Training model...")
    train(model, dataloader, num_epochs)

    print("Training complete.")


if __name__ == "__main__":
    main()
