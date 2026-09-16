from torch import nn
from .common import SequenceEncoder


class AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.encoder=SequenceEncoder(cfg["input_dim"],cfg["hidden_dim"],cfg["transformer_layers"],cfg["attention_heads"],cfg["ffn_dim"],cfg["dropout"],cfg["max_tokens"])
    def forward(self,x,mask): return self.encoder(x,mask)
