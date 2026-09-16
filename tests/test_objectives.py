import torch
from merge_emotion.objectives.affect_aware import affect_distance_weights, affect_aware_contrastive_loss
from merge_emotion.objectives.contrastive import symmetric_contrastive_loss


def test_contrastive_loss_prefers_matched_pairs():
    x = torch.eye(4)
    good = symmetric_contrastive_loss(x, x, 0.1)
    bad = symmetric_contrastive_loss(x, x.flip(0), 0.1)
    assert good < bad


def test_affect_weights_downweight_same_quadrant_negatives():
    labels = torch.tensor([0, 0, 2])
    w = affect_distance_weights(labels, epsilon=0.1, gamma=1.0)
    assert torch.isclose(w[0, 1], torch.tensor(0.1))
    assert w[0, 2] > w[0, 1]


def test_affect_aware_loss_is_finite():
    a = torch.randn(8, 16)
    l = torch.randn(8, 16)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    assert torch.isfinite(affect_aware_contrastive_loss(a, l, y))
