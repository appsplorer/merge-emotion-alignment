from __future__ import annotations
from typing import Iterable, Sequence
import torch


def flatten_grads(grads: Sequence, params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    pieces = []
    for g, p in zip(grads, params):
        pieces.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
    return torch.cat(pieces) if pieces else torch.empty(0)


def assign_flat_grad(params: Sequence[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset:offset+n].view_as(p).clone()
        offset += n
