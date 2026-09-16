from __future__ import annotations

import torch
import torch.nn.functional as F

QUADRANT_COORDINATES = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])


def affect_distance_weights(labels: torch.Tensor, epsilon: float = 0.1, gamma: float = 1.0) -> torch.Tensor:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    coords = QUADRANT_COORDINATES.to(labels.device)[labels]
    distances = torch.cdist(coords, coords, p=2) / (2.0 * (2.0 ** 0.5))
    weights = epsilon + (1.0 - epsilon) * distances.pow(gamma)
    weights.fill_diagonal_(1.0)
    return weights


def affect_aware_contrastive_loss(audio_repr, lyrics_repr, labels, temperature=0.07, epsilon=0.1, gamma=1.0):
    a = F.normalize(audio_repr, p=2, dim=-1)
    l = F.normalize(lyrics_repr, p=2, dim=-1)
    logits = a @ l.t() / temperature
    weights = affect_distance_weights(labels, epsilon, gamma)
    eye = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    log_w = torch.log(weights.clamp_min(1e-12))
    weighted_logits = torch.where(eye, logits, logits + log_w)
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(weighted_logits, targets) + F.cross_entropy(weighted_logits.t(), targets))
