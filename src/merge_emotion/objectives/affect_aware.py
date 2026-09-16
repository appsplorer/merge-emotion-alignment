from __future__ import annotations
import math
import torch
import torch.nn.functional as F

QUADRANT_COORDS = torch.tensor([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]])


def affect_distance_weights(labels: torch.Tensor, epsilon=0.1, gamma=1.0):
    coords = QUADRANT_COORDS.to(labels.device)[labels]
    d = torch.cdist(coords, coords, p=2) / (2.0 * math.sqrt(2.0))
    return epsilon + (1.0 - epsilon) * d.pow(gamma)


def _directional(logits, weights):
    n = logits.shape[0]
    eye = torch.eye(n, device=logits.device, dtype=torch.bool)
    weighted_neg = torch.exp(logits) * weights.masked_fill(eye, 0.0)
    numerator = torch.exp(logits.diag())
    denominator = numerator + weighted_neg.sum(dim=1)
    return (-torch.log(numerator / denominator.clamp_min(1e-12))).mean()


def affect_aware_contrastive_loss(audio_repr, lyrics_repr, labels, temperature=0.07, epsilon=0.1, gamma=1.0):
    a = F.normalize(audio_repr, dim=-1); l = F.normalize(lyrics_repr, dim=-1)
    logits = a @ l.T / temperature
    w = affect_distance_weights(labels, epsilon, gamma)
    return 0.5 * (_directional(logits, w) + _directional(logits.T, w.T))
