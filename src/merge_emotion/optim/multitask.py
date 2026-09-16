from __future__ import annotations
import torch


def pcgrad_two_task(g1: torch.Tensor, g2: torch.Tensor, eps=1e-12):
    a, b = g1.clone(), g2.clone()
    dot = torch.dot(a, b)
    if dot < 0:
        a = a - dot / b.pow(2).sum().clamp_min(eps) * b
        b = b - dot / g1.pow(2).sum().clamp_min(eps) * g1
    return 0.5 * (a + b)


def cagrad_two_task(g1: torch.Tensor, g2: torch.Tensor, c=0.5, grid_points=101, eps=1e-12):
    """Two-task conflict-averse gradient via deterministic 1-D simplex search.

    This is an explicit two-task implementation used as a comparator, not a claim of a
    reimplementation of every optimization detail from the original CAGrad codebase.
    """
    g0 = 0.5 * (g1 + g2)
    best_w, best_obj = 0.5, None
    scale = c * torch.linalg.vector_norm(g0).item()
    for k in range(grid_points):
        w = k / float(grid_points - 1)
        gw = w * g1 + (1.0 - w) * g2
        obj = torch.dot(gw, g0).item() + scale * torch.linalg.vector_norm(gw).item()
        if best_obj is None or obj < best_obj:
            best_obj, best_w = obj, w
    gw = best_w * g1 + (1.0 - best_w) * g2
    norm = torch.linalg.vector_norm(gw).clamp_min(eps)
    return g0 + scale * gw / norm
