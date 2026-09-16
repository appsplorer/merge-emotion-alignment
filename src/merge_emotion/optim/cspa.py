from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    directional_margin: float
    constraint_active: bool
    feasible: bool


class SecantCurvatureEstimator:
    """Empirical local smoothness proxy from successive primary gradients.

    This is deliberately labelled an estimate, not a certified Lipschitz bound.
    A conditional descent-lemma statement is valid only if the supplied beta is
    in fact an upper bound and the update follows the raw gradient direction.
    """

    def __init__(self, safety_factor=1.5, beta_min=0.0, beta_max=1e4, eps=1e-8):
        self.safety_factor = float(safety_factor)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.eps = float(eps)
        self.prev_theta: Optional[torch.Tensor] = None
        self.prev_grad: Optional[torch.Tensor] = None
        self.beta = self.beta_min

    def update(self, theta: torch.Tensor, grad: torch.Tensor) -> float:
        if self.prev_theta is not None and self.prev_grad is not None:
            denom = torch.linalg.vector_norm(theta - self.prev_theta).item()
            if denom > self.eps:
                estimate = self.safety_factor * torch.linalg.vector_norm(grad - self.prev_grad).item() / denom
                self.beta = min(self.beta_max, max(self.beta_min, estimate))
        self.prev_theta = theta.detach().clone()
        self.prev_grad = grad.detach().clone()
        return float(self.beta)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "prev_theta": self.prev_theta,
            "prev_grad": self.prev_grad,
            "beta": float(self.beta),
        }

    def load_state_dict(self, state: Dict[str, Any], device=None) -> None:
        self.prev_theta = state.get("prev_theta")
        self.prev_grad = state.get("prev_grad")
        if device is not None:
            if self.prev_theta is not None:
                self.prev_theta = self.prev_theta.to(device)
            if self.prev_grad is not None:
                self.prev_grad = self.prev_grad.to(device)
        self.beta = float(state.get("beta", self.beta_min))


class CSPAController:
    """Primary-emotion-preserving adaptive alignment controller.

    The controller caps the auxiliary alignment coefficient by (i) a relative
    gradient-norm limit and (ii) a beta-smooth descent-lemma constraint. With
    empirical secant beta + AdamW this is a safeguard/diagnostic, not a formal
    certificate. The logged directional margin is exact for the local raw
    gradient direction and is useful for auditing negative-transfer pressure.
    """

    def __init__(self, lambda_max=1.0, kappa=0.8, rho=0.5, ema_beta=0.9, eps=1e-8):
        if not 0.0 < float(kappa) < 1.0:
            raise ValueError("kappa must be in (0, 1)")
        self.lambda_max = float(lambda_max)
        self.kappa = float(kappa)
        self.rho = float(rho)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        self._ema: Optional[float] = None

    def _curvature_bound(self, ge2, ga2, dot, lr, beta):
        # Sufficient beta-smooth descent-lemma constraint for raw SGD:
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
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return 0.0, False
        root = (-b + math.sqrt(max(0.0, disc))) / (2.0 * a)
        return max(0.0, root), True

    def choose(self, ge: torch.Tensor, ga: torch.Tensor, lr: float, beta: float) -> CSPAState:
        ne = torch.linalg.vector_norm(ge).item()
        na = torch.linalg.vector_norm(ga).item()
        dot = torch.dot(ge, ga).item()
        cos = dot / max(ne * na, self.eps)
        ge2, ga2 = ne * ne, na * na
        l_norm = self.rho * ne / max(na, self.eps) if na > self.eps else self.lambda_max
        l_curv, feasible = self._curvature_bound(ge2, ga2, dot, float(lr), float(beta))
        raw = min(self.lambda_max, l_norm, l_curv)
        if self._ema is None:
            used = raw
        else:
            smoothed = self.ema_beta * self._ema + (1.0 - self.ema_beta) * raw
            used = min(smoothed, raw)
        self._ema = float(used)
        directional_margin = ge2 + used * dot - self.kappa * ge2
        return CSPAState(
            lambda_raw=float(raw),
            lambda_used=float(used),
            lambda_curvature=float(l_curv),
            lambda_norm=float(l_norm),
            grad_cosine=float(cos),
            emotion_grad_norm=float(ne),
            alignment_grad_norm=float(na),
            beta=float(beta),
            directional_margin=float(directional_margin),
            constraint_active=bool(used < self.lambda_max - 1e-10),
            feasible=bool(feasible),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {"ema": self._ema}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        value = state.get("ema")
        self._ema = None if value is None else float(value)
