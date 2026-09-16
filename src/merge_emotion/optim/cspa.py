from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CSPAState:
    lambda_raw: float
    lambda_used: float
    lambda_curvature: float
    lambda_norm: float
    grad_cosine: float
    emotion_grad_norm: float
    alignment_grad_norm: float
    beta: float
    constraint_active: bool
    feasible: bool


class SecantCurvatureEstimator:
    """Empirical local smoothness proxy from successive primary gradients.

    This is not a certified global Lipschitz bound. A safety factor is applied and the
    resulting CSPA condition must be reported as an empirical safeguard unless a valid
    upper bound is independently established.
    """
    def __init__(self, safety_factor=1.5, beta_min=0.0, beta_max=1e4, eps=1e-8):
        self.safety_factor = safety_factor; self.beta_min = beta_min; self.beta_max = beta_max; self.eps = eps
        self.prev_theta = None; self.prev_grad = None

    def update(self, theta: torch.Tensor, grad: torch.Tensor) -> float:
        beta = self.beta_min
        if self.prev_theta is not None:
            denom = torch.linalg.vector_norm(theta - self.prev_theta).item()
            if denom > self.eps:
                beta = self.safety_factor * torch.linalg.vector_norm(grad - self.prev_grad).item() / denom
        self.prev_theta = theta.detach().clone(); self.prev_grad = grad.detach().clone()
        return float(min(self.beta_max, max(self.beta_min, beta)))


class CSPAController:
    def __init__(self, lambda_max=1.0, kappa=0.8, rho=0.5, ema_beta=0.9, eps=1e-8):
        self.lambda_max = float(lambda_max); self.kappa = float(kappa); self.rho = float(rho); self.ema_beta = float(ema_beta); self.eps = float(eps); self._ema = None

    def _curvature_bound(self, ge2, ga2, dot, lr, beta):
        # Sufficient descent-lemma constraint:
        # .5*beta*lr*||ga||^2*l^2 + (beta*lr-1)<ge,ga>*l
        # + (.5*beta*lr-(1-kappa))*||ge||^2 <= 0.
        a = 0.5 * beta * lr * ga2
        b = (beta * lr - 1.0) * dot
        c = (0.5 * beta * lr - (1.0 - self.kappa)) * ge2
        if c > 0:
            return 0.0, False
        if abs(a) < self.eps:
            if b <= 0:
                return self.lambda_max, True
            return max(0.0, -c / max(b, self.eps)), True
        disc = max(0.0, b*b - 4.0*a*c)
        root = (-b + disc ** 0.5) / (2.0*a)
        return max(0.0, root), True

    def choose(self, ge: torch.Tensor, ga: torch.Tensor, lr: float, beta: float) -> CSPAState:
        ne = torch.linalg.vector_norm(ge).item(); na = torch.linalg.vector_norm(ga).item(); dot = torch.dot(ge, ga).item()
        cos = dot / max(ne * na, self.eps)
        l_norm = self.rho * ne / max(na, self.eps)
        l_curv, feasible = self._curvature_bound(ne*ne, na*na, dot, float(lr), float(beta))
        raw = min(self.lambda_max, l_norm, l_curv)
        used = raw if self._ema is None else self.ema_beta * self._ema + (1.0 - self.ema_beta) * raw
        used = min(used, self.lambda_max, l_norm, l_curv)
        self._ema = used
        active = used < self.lambda_max - 1e-10
        return CSPAState(raw, used, l_curv, l_norm, cos, ne, na, float(beta), active, feasible)
