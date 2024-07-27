'''
[24/06/19 patch note]
- Adding 'PositionalEncoding' Class
- Modifying 'PositionWiseFeedForward' method

[24/06/21 patch note]
- Adding 'ComplexLinear' Class
- Modifying nn.Linear to ComplexLinear

[24/06/23 patch note]
- Adding File Opening Code
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import Tensor


class ComplexLinear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    real_weight: Tensor
    imag_weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.real_weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.imag_weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.real_bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.imag_bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('real_bias', None)
            self.register_parameter('imag_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.real_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.imag_weight, a=math.sqrt(5))
        if self.real_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.real_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.real_bias, -bound, bound)
            nn.init.uniform_(self.imag_bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        input_real = input.real
        input_imag = input.imag

        real_output = torch.matmul(input_real, self.real_weight.T) - torch.matmul(input_imag, self.imag_weight.T)
        imag_output = torch.matmul(input_real, self.imag_weight.T) + torch.matmul(input_imag, self.real_weight.T)

        if self.real_bias is not None:
            real_output += self.real_bias
            imag_output += self.imag_bias

        return torch.complex(real_output, imag_output)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.real_bias is not None}'

class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_block) for _ in range(n_layer)])

    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out

class EncoderBlock(nn.Module):

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

class ResidualConnectionLayer(nn.Module):

    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out + x
        return x

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = deepcopy(qkv_fc)
        self.k_fc = deepcopy(qkv_fc)
        self.v_fc = deepcopy(qkv_fc)
        self.out_fc = out_fc

    def forward(self, query, key, value, mask=None):
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

class PositionalEncoding(nn.Module):

    def __init__(self, d_embed = 512, max_len = 256, device = torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding_requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(d_embed) *  -(math.log(10000.0) / d_embed))
        encoding[:, :] = torch.exp(1j* position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out

class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1 # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2 # (d_ff, d_embed)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.zrelu(out)
        out = self.fc2(out)
        return out

    def zrelu(self, z): #z : tensor 활성화 함수 1
        real_parts = z.real
        imag_parts = z.imag

        mask = (real_parts > 0) & (imag_parts > 0)

        filtered_tensor = torch.where(mask, z, torch.zeros_like(z))

        return filtered_tensor

### Hyperparameter ###
d_embed = 512
d_model = 512
h = 8
d_ff = 2048
n_layer = 6
batch_size = 64
num_epochs = 100

### Model Structure ###
attention = MultiHeadAttentionLayer(d_model = d_model, h = h, qkv_fc = ComplexLinear(d_embed, d_model), out_fc = ComplexLinear(d_model, d_embed))
position_ff = PositionWiseFeedForwardLayer(fc1 = ComplexLinear(d_embed, d_ff), fc2 = ComplexLinear(d_ff, d_embed))

encoder_block = EncoderBlock(self_attention = attention, position_ff = position_ff)

model = Encoder(encoder_block, n_layer)

### Random Data Generate ###
'''
x_train = torch.complex(torch.rand(8551,45,512),torch.rand(8551,45,512))
y_train = torch.randint(0, 2, (8551,)).float()

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
'''

### Data Upload ###
import pandas as pd

CoLA_raw_df = pd.read_csv('train.tsv', sep='\t')
CoLA_raw_df.columns = ['Dummy', 'Label', 'Dummy2', 'Sentence']
CoLA_raw_df.drop(columns=['Dummy', 'Dummy2'], inplace=True)

### Data tokenizing ###
import torchtext
from torchtext.data import get_tokenizer
import numpy as np
from torch.nn.utils.rnn import pad_sequence

tokenizer = get_tokenizer('basic_english')
CoLA_raw_df['Tokenized_sentence'] = CoLA_raw_df['Sentence'].apply(tokenizer)
CoLA_raw_df.drop(columns=['Sentence'], inplace = True)
max_len = max(len(sentences) for sentences in CoLA_raw_df['Tokenized_sentence']) + 1

vocab = set(word for sentence in CoLA_raw_df['Tokenized_sentence'] for word in sentence)
word2idx = {word : idx for idx, word in enumerate(vocab)}

embedding_layer = nn.Embedding(len(vocab), d_embed)

def get_word_embedding(word, word2idx, embedding_layer):
    idx = word2idx[word]
    idx_tensor = torch.tensor([idx], dtype=torch.long)
    embedding_vector = embedding_layer(idx_tensor).detach()
    return embedding_vector

def embed_sentence(sentence, word2idx, embedding_layer):
    return torch.cat([get_word_embedding(word, word2idx, embedding_layer) for word in sentence])

CoLA_raw_df['embedding_sentence'] = CoLA_raw_df['Tokenized_sentence'].apply(
    lambda tokens: embed_sentence(tokens, word2idx, embedding_layer)
)

embedded_sentences = list(CoLA_raw_df['embedding_sentence'])
padded_embeddings = pad_sequence(embedded_sentences, batch_first=True)

x_train = padded_embeddings.clone().detach().to(torch.cfloat)
y_train = torch.tensor(CoLA_raw_df['Label'].values, dtype=torch.float32)

x_train = PositionalEncoding(d_embed=d_embed, max_len = max_len).forward(x_train)

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

### Opt, Cri ###
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        src_mask = None  # Assuming no mask for simplicity
        output = model(batch_x, src_mask)
        output = output.mean(dim=1)
        output = ComplexLinear(d_model, 1)(output)
        output = abs(output)
        output = torch.sigmoid(output)  # Apply sigmoid for binary classification
        output = output.squeeze(-1)

        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete.")