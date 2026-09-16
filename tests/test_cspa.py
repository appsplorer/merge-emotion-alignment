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
