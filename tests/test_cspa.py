import torch
from merge_emotion.optim.cspa import CSPAController


def test_cspa_reduces_weight_for_strong_conflict():
    cspa = CSPAController(lambda_max=1.0, kappa=0.8, rho=1.0, ema_beta=0.0)
    ge = torch.tensor([1.0, 0.0])
    ga = torch.tensor([-10.0, 0.0])
    state = cspa.choose(ge, ga, lr=1e-3, beta=0.0)
    assert 0.0 <= state.lambda_used < 1.0
    assert state.grad_cosine < 0.0


def test_cspa_allows_large_weight_when_gradients_agree():
    cspa = CSPAController(lambda_max=1.0, kappa=0.8, rho=10.0, ema_beta=0.0)
    ge = torch.tensor([1.0, 0.0])
    ga = torch.tensor([0.1, 0.0])
    state = cspa.choose(ge, ga, lr=1e-3, beta=0.0)
    assert state.lambda_used == 1.0


def test_cspa_conditional_descent_bound_on_exact_quadratic():
    beta = 2.0
    lr = 0.05
    kappa = 0.8
    theta = torch.tensor([0.7, -0.4], dtype=torch.float64)
    ge = beta * theta
    ga = torch.tensor([-1.2, 0.9], dtype=torch.float64)
    controller = CSPAController(lambda_max=1.0, kappa=kappa, rho=10.0, ema_beta=0.0)
    state = controller.choose(ge, ga, lr=lr, beta=beta)
    direction = ge + state.lambda_used * ga
    updated = theta - lr * direction
    before = 0.5 * beta * torch.dot(theta, theta)
    after = 0.5 * beta * torch.dot(updated, updated)
    guaranteed_upper = before - kappa * lr * torch.dot(ge, ge)
    assert after <= guaranteed_upper + 1e-10
    assert state.feasible
    assert state.directional_margin >= -1e-10
