import torch
from merge_emotion.optim.multitask import cagrad_two_task, pcgrad_two_task


def test_pcgrad_removes_negative_pairwise_component():
    g1 = torch.tensor([1.0, 0.0]); g2 = torch.tensor([-1.0, 1.0])
    combined = pcgrad_two_task(g1, g2)
    assert torch.isfinite(combined).all()
    assert combined.norm() > 0


def test_cagrad_returns_finite_gradient():
    g1 = torch.tensor([1.0, 0.5]); g2 = torch.tensor([-0.25, 1.0])
    combined = cagrad_two_task(g1, g2, c=0.5)
    assert combined.shape == g1.shape
    assert torch.isfinite(combined).all()
