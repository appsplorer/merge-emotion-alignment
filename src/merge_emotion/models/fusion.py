from __future__ import annotations
import torch
from torch import nn


class GatedCrossModalFusion(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, fusion_layers, ffn_dim):
        super().__init__()
        self.a_to_l = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.l_to_a = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.gate_a = nn.Linear(hidden_dim * 2, hidden_dim); self.gate_l = nn.Linear(hidden_dim * 2, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, ffn_dim, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(layer, num_layers=fusion_layers, norm=nn.LayerNorm(hidden_dim))

    def forward(self, audio, lyrics, audio_mask, lyrics_mask):
        a_ctx, _ = self.a_to_l(audio, lyrics, lyrics, key_padding_mask=~lyrics_mask.bool())
        l_ctx, _ = self.l_to_a(lyrics, audio, audio, key_padding_mask=~audio_mask.bool())
        ga = torch.sigmoid(self.gate_a(torch.cat([audio, a_ctx], dim=-1))); gl = torch.sigmoid(self.gate_l(torch.cat([lyrics, l_ctx], dim=-1)))
        a = ga * audio + (1-ga) * a_ctx; l = gl * lyrics + (1-gl) * l_ctx
        tokens = torch.cat([a, l], dim=1); mask = torch.cat([audio_mask, lyrics_mask], dim=1)
        tokens = self.fusion(tokens, src_key_padding_mask=~mask.bool())
        w = mask.float().unsqueeze(-1)
        return (tokens*w).sum(1)/w.sum(1).clamp_min(1.0)
