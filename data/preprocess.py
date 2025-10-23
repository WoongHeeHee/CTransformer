import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
import numpy as np
from model.positional_encoding import PositionalEncoding


def load_cola_data(path='data/train.tsv'):
    """
    Load CoLA dataset from TSV file
    """
    CoLA_raw_df = pd.read_csv(path, sep='\t')
    CoLA_raw_df.columns = ['Dummy', 'Label', 'Dummy2', 'Sentence']
    CoLA_raw_df.drop(columns=['Dummy', 'Dummy2'], inplace=True)
    return CoLA_raw_df


def tokenize_sentences(df):
    """
    Tokenize sentences using basic English tokenizer
    """
    tokenizer = get_tokenizer('basic_english')
    df['Tokenized_sentence'] = df['Sentence'].apply(tokenizer)
    df.drop(columns=['Sentence'], inplace=True)
    return df


def create_vocabulary(df):
    """
    Create vocabulary from tokenized sentences
    """
    vocab = set(word for sentence in df['Tokenized_sentence'] for word in sentence)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2idx


def create_embedding_layer(vocab_size, d_embed):
    """
    Create embedding layer
    """
    return nn.Embedding(vocab_size, d_embed)


def get_word_embedding(word, word2idx, embedding_layer):
    """
    Get embedding for a single word
    """
    idx = word2idx[word]
    idx_tensor = torch.tensor([idx], dtype=torch.long)
    embedding_vector = embedding_layer(idx_tensor).detach()
    return embedding_vector


def embed_sentence(sentence, word2idx, embedding_layer):
    """
    Embed a complete sentence
    """
    return torch.cat([get_word_embedding(word, word2idx, embedding_layer) for word in sentence])


def preprocess_data(path='data/train.tsv', d_embed=512, batch_size=64):
    """
    Complete data preprocessing pipeline
    """
    # Load data
    df = load_cola_data(path)
    
    # Tokenize
    df = tokenize_sentences(df)
    max_len = max(len(sentences) for sentences in df['Tokenized_sentence']) + 1
    
    # Create vocabulary
    vocab, word2idx = create_vocabulary(df)
    
    # Create embedding layer
    embedding_layer = create_embedding_layer(len(vocab), d_embed)
    
    # Embed sentences
    df['embedding_sentence'] = df['Tokenized_sentence'].apply(
        lambda tokens: embed_sentence(tokens, word2idx, embedding_layer)
    )
    
    # Pad sequences
    embedded_sentences = list(df['embedding_sentence'])
    padded_embeddings = pad_sequence(embedded_sentences, batch_first=True)
    
    # Convert to complex tensors
    x_train = padded_embeddings.clone().detach().to(torch.cfloat)
    y_train = torch.tensor(df['Label'].values, dtype=torch.float32)
    
    # Add positional encoding
    x_train = PositionalEncoding(d_embed=d_embed, max_len=max_len).forward(x_train)
    
    # Create dataset and dataloader
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, word2idx, embedding_layer
