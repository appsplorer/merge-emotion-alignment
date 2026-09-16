from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize

from merge_emotion.analysis.statistics import seed_summary
from merge_emotion.engine.checkpoint import config_fingerprint, load_checkpoint, save_checkpoint
from merge_emotion.objectives.affect_aware import affect_aware_contrastive_loss
from merge_emotion.optim.multitask import cagrad_two_task


ROOT = Path(__file__).resolve().parents[1]


def test_training_entrypoint_has_hard_test_firewall():
    source = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert 'MergeFeatureDataset(manifest, "test"' not in source
    assert "test_loader" not in source
    assert 'run_dir / "predictions.csv"' not in source


def test_final_evaluation_entrypoint_exists_and_requires_selection_manifest():
    path = ROOT / "scripts/final_evaluate.py"
    assert path.exists()
    source = path.read_text(encoding="utf-8")
    assert "--selection-manifest" in source
    assert '"test"' in source


def test_checkpoint_roundtrip_includes_trainer_state_and_config_fingerprint(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {"project": {"seed": 42}, "training": {"max_epochs": 5}}
    trainer_state = {"patience": 2}
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, None, None, 3, 17, 0.7, config, trainer_state=trainer_state)
    payload = load_checkpoint(path, model, optimizer, restore_rng=False)
    assert payload["trainer_state"]["patience"] == 2
    assert payload["config_fingerprint"] == config_fingerprint(config)


def test_affect_aware_loss_is_finite_for_extreme_logits():
    # Cosine / tiny temperature creates logits ~ +/-1000. A direct exp(logits)
    # implementation overflows, while log-sum-exp / cross-entropy remains stable.
    audio = torch.eye(4)
    lyrics = torch.eye(4)
    labels = torch.tensor([0, 1, 2, 3])
    loss = affect_aware_contrastive_loss(audio, lyrics, labels, temperature=1e-3, epsilon=0.1, gamma=1.0)
    assert torch.isfinite(loss)


def _official_two_task_cagrad_reference(g1: torch.Tensor, g2: torch.Tensor, alpha: float) -> torch.Tensor:
    grads = torch.stack([g1, g2], dim=0).double()
    g0 = grads.mean(0)
    A = (grads @ grads.t()).cpu().numpy()
    b = np.ones(2, dtype=np.float64) / 2.0
    c = alpha * float(g0.norm())

    def obj(x):
        x = np.asarray(x, dtype=np.float64)
        linear = float((x.reshape(1, 2) @ A @ b.reshape(2, 1))[0, 0])
        quadratic = float((x.reshape(1, 2) @ A @ x.reshape(2, 1))[0, 0])
        return linear + c * np.sqrt(quadratic + 1e-8)

    res = minimize(
        obj,
        b.copy(),
        bounds=((0.0, 1.0), (0.0, 1.0)),
        constraints=({"type": "eq", "fun": lambda x: 1.0 - float(np.sum(x))},),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert res.success
    w = torch.tensor(res.x, dtype=grads.dtype)
    gw = (grads * w.view(-1, 1)).sum(0)
    lam = c / (float(gw.norm()) + 1e-4)
    out = (g0 + lam * gw) / (1.0 + lam)
    return out.to(dtype=g1.dtype)


@pytest.mark.parametrize(
    "g1,g2",
    [
        (torch.tensor([1.0, 0.5, -0.2]), torch.tensor([-0.25, 1.0, 0.7])),
        (torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])),
        (torch.tensor([0.2, -0.4, 1.2]), torch.tensor([0.6, 0.7, -0.3])),
    ],
)
def test_cagrad_matches_official_exact_two_task_objective(g1, g2):
    actual = cagrad_two_task(g1, g2, c=0.5)
    expected = _official_two_task_cagrad_reference(g1, g2, alpha=0.5)
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3), (actual, expected)


def test_seed_summary_includes_bootstrap_ci_and_count():
    frame = pd.DataFrame(
        {
            "experiment": ["a"] * 5,
            "seed": [1, 2, 3, 4, 5],
            "macro_f1": [0.60, 0.62, 0.61, 0.64, 0.63],
        }
    )
    summary = seed_summary(frame, metrics=["macro_f1"], bootstrap_samples=500, seed=7)
    row = summary.iloc[0]
    assert row["n"] == 5
    assert row["macro_f1_ci_low"] <= row["macro_f1_mean"] <= row["macro_f1_ci_high"]


def test_all_slurm_arrays_are_serialized_to_one_gpu():
    for path in sorted((ROOT / "slurm").glob("*.sbatch")):
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            if line.startswith("#SBATCH --array="):
                assert line.rstrip().endswith("%1"), f"{path}: {line}"


def test_final_evaluation_skips_already_completed_checkpoint_hash():
    source = (ROOT / "scripts/final_evaluate.py").read_text(encoding="utf-8")
    assert "checkpoint_sha256" in source
    assert "already evaluated" in source


def test_optuna_trials_are_a_total_target_not_added_on_every_resume():
    source = (ROOT / "scripts/tune.py").read_text(encoding="utf-8")
    assert "remaining_trials" in source
    assert "TrialState.COMPLETE" in source
