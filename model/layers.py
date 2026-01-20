import torch
import torch.nn as nn
from model.attention import MultiHeadSelfAttention

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(config.d_model, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.FFN(x)
    
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps:float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        x_normalised = (x-x_mean) / torch.sqrt(x_var + self.eps)
        return self.gamma*x_normalised+self.beta
    
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_layer = MultiHeadSelfAttention(config)
        self.ffn = FeedForwardNetwork(config)
        self.ln1 = LayerNorm(config.d_model)
        self.ln2 = LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, src_mask):
        res = x
        x = self.ln1(x)
        attn_output = self.dropout(self.attn_layer(q=x, k=x, v=x, mask=src_mask))
        x_attn = res + attn_output

        res_attn = x_attn
        x_attn = self.ln2(x_attn)
        x_ffn = res_attn + self.ffn(x_attn)

        return x_ffn
