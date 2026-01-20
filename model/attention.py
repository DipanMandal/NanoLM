import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.num_heads == 0

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_head = config.d_model // config.num_heads

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        self.W_o = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        query = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        attention_score = F.softmax(torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_head), dim = -1)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        
        attention_score = self.dropout(attention_score)
        out = torch.matmul(attention_score, value)

        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        out = self.W_o(out)

        return out

if __name__ == "__main__":
    from model.config import NanoLMConfig

    config = NanoLMConfig()
    d_model = config.d_model
    seq_len = config.sequence_length

    vec = torch.randn(2, seq_len, d_model)
    attn_module = MultiHeadSelfAttention(config)
    out = attn_module(vec, vec, vec)
    print(out.shape)


        

