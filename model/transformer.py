import torch
import torch.nn as nn
from model.embeddings import TransformerEmbedding
from model.layers import EncoderLayer, LayerNorm

class EncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = TransformerEmbedding(config)
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range (config.n_encoder_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.ln_final = LayerNorm(config.d_model)
        self.apply(self._init_weights)
        self.print_parameter_count()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def print_parameter_count(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Trainable Params :: {params:,}")

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)

        return self.ln_final(x)
    
if __name__ == "__main__":
    from model.config import NanoLMConfig

    config = NanoLMConfig()
    enc_model = EncoderModel(config)