from __future__ import annotations
import torch
from torch import nn


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, heads, ffn_dim, dropout, max_tokens):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, ffn_dim, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(hidden_dim))

    def forward(self, x, mask):
        h = self.proj(x) + self.pos[:, :x.size(1)]
        h = self.encoder(h, src_key_padding_mask=~mask.bool())
        weights = mask.float().unsqueeze(-1)
        pooled = (h * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return h, pooled
