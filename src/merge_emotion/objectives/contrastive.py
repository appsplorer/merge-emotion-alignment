from __future__ import annotations
import torch
import torch.nn.functional as F


def symmetric_contrastive_loss(audio_repr, lyrics_repr, temperature=0.07):
    a = F.normalize(audio_repr, dim=-1); l = F.normalize(lyrics_repr, dim=-1)
    logits = a @ l.T / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
