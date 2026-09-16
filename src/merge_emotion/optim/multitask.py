from __future__ import annotations

import torch


def pcgrad_two_task(g_primary: torch.Tensor, g_aux: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Deterministic two-task PCGrad specialization.

    For two objectives the random task ordering in canonical PCGrad reduces to
    projecting each gradient against the other when their dot product is negative.
    The mean keeps update scale comparable to an average two-task gradient.
    """
    a, b = g_primary.clone(), g_aux.clone()
    dot = torch.dot(a, b)
    if dot < 0:
        a = a - dot / b.pow(2).sum().clamp_min(eps) * b
        b = b - dot / g_primary.pow(2).sum().clamp_min(eps) * g_primary
    return 0.5 * (a + b)


def _cagrad_objective_two_task(w: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor, c: torch.Tensor, eps: float) -> torch.Tensor:
    gw = w * g1 + (1.0 - w) * g2
    g0 = 0.5 * (g1 + g2)
    return torch.dot(gw, g0) + c * torch.sqrt(torch.dot(gw, gw) + eps)


def cagrad_two_task(
    g_primary: torch.Tensor,
    g_aux: torch.Tensor,
    c: float = 0.5,
    iterations: int = 72,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Exact-objective two-task CAGrad via device-local scalar search.

    This minimizes the same 2-task simplex dual objective used by the official
    CAGrad exact solver. The bounded search stays on the gradient device and
    avoids CPU/SciPy synchronization inside the training loop.
    """
    if float(c) < 0.0:
        raise ValueError("c must be non-negative")
    g1 = g_primary
    g2 = g_aux
    g0 = 0.5 * (g1 + g2)
    c_tensor = g0.new_tensor(float(c)) * torch.linalg.vector_norm(g0)
    left = g0.new_tensor(0.0)
    right = g0.new_tensor(1.0)
    for _ in range(int(iterations)):
        third = (right - left) / 3.0
        x1 = left + third
        x2 = right - third
        f1 = _cagrad_objective_two_task(x1, g1, g2, c_tensor, eps)
        f2 = _cagrad_objective_two_task(x2, g1, g2, c_tensor, eps)
        choose_left = f1 <= f2
        right = torch.where(choose_left, x2, right)
        left = torch.where(choose_left, left, x1)
    w = 0.5 * (left + right)
    gw = w * g1 + (1.0 - w) * g2
    lam = c_tensor / (torch.linalg.vector_norm(gw) + 1e-4)
    return (g0 + lam * gw) / (1.0 + lam)
