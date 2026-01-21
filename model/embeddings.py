import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_length: int):
        super().__init__()
        pe = torch.zeros(sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0))/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
class TransformerEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pe = PositionalEncoding(config.d_model, config.sequence_length)
        self.dropout = nn.Dropout(config.dropout)
        self.scaling = torch.sqrt(torch.tensor(config.d_model))

    def forward(self, x):
        out = self.word_embeddings(x)
        out = self.pe(out)
        return self.dropout(out)