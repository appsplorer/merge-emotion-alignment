from __future__ import annotations

import pytest
import pandas as pd


def _load_feature():
    try:
        from merge_emotion.analysis.sensitivity import (
            ARCHITECTURE_SEEDS,
            architecture_variants,
            build_training_command,
            summarize_architecture_results,
            summarize_component_ablation,
            summarize_ablation_effects,
        )
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.fail(
            "Architecture-sensitivity implementation has not been created yet: %s" % exc
        )
    return (
        ARCHITECTURE_SEEDS,
        architecture_variants,
        build_training_command,
        summarize_architecture_results,
        summarize_component_ablation,
        summarize_ablation_effects,
    )


def test_architecture_sensitivity_contract():
    (
        seeds,
        variants_fn,
        _,
        _,
        _,
        _,
    ) = _load_feature()

    variants = variants_fn()

    assert tuple(seeds) == (42, 123, 777)
    assert len(variants) == 9

    assert variants["base"] == {
        "hidden_dim": 256,
        "modality_layers": 2,
        "fusion_layers": 2,
        "heads": 4,
        "ffn_dim": 512,
    }

    expected = {
        "base",
        "hidden_128",
        "hidden_384",
        "encoder_1",
        "encoder_3",
        "fusion_1",
        "fusion_3",
        "heads_2",
        "heads_8",
    }
    assert set(variants) == expected

    for spec in variants.values():
        assert spec["hidden_dim"] % spec["heads"] == 0


def test_architecture_training_command_is_validation_only():
    (
        _,
        variants_fn,
        build_command,
        _,
        _,
        _,
    ) = _load_feature()

    cmd = build_command(
        variant_name="base",
        spec=variants_fn()["base"],
        seed=42,
    )

    joined = " ".join(cmd)

    assert "scripts/train.py" in joined
    assert "--require-cuda" in cmd
    assert "--resume" in cmd
    assert "auto" in cmd

    assert "experiment.stage=architecture_sensitivity" in joined
    assert "contrastive.lambda_align=0.2" in joined
    assert "project.seed=42" in joined

    # train.py evaluates train + validation only.
    assert "final_evaluate.py" not in joined
    assert "--split=test" not in joined
    assert "--split test" not in joined


def test_architecture_summary_groups_three_seeds():
    (
        _,
        _,
        _,
        summarize_arch,
        _,
        _,
    ) = _load_feature()

    frame = pd.DataFrame(
        {
            "variant": ["base"] * 3 + ["hidden_128"] * 3,
            "seed": [42, 123, 777] * 2,
            "macro_f1": [0.80, 0.82, 0.81, 0.78, 0.79, 0.77],
            "accuracy": [0.81, 0.83, 0.82, 0.79, 0.80, 0.78],
            "alignment_gap": [0.10, 0.11, 0.12, 0.08, 0.09, 0.10],
        }
    )

    summary = summarize_arch(frame)

    base = summary.loc[summary["variant"] == "base"].iloc[0]
    assert int(base["n"]) == 3
    assert base["macro_f1_mean"] == pytest.approx(0.81)
    assert "macro_f1_sd" in summary.columns


def test_component_ablation_and_interaction_are_paired_by_seed():
    (
        _,
        _,
        _,
        _,
        summarize_ablation,
        summarize_effects,
    ) = _load_feature()

    rows = []
    values = {
        "multimodal_fixed": [0.80, 0.81, 0.82],
        "affect_only": [0.79, 0.80, 0.81],
        "cspa_only": [0.78, 0.79, 0.80],
        "cspa_affect": [0.83, 0.84, 0.85],
    }

    for experiment, scores in values.items():
        for seed, score in zip([42, 123, 777], scores):
            rows.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "macro_f1": score,
                    "accuracy": score,
                }
            )

    frame = pd.DataFrame(rows)

    table = summarize_ablation(frame)
    effects = summarize_effects(frame)

    assert set(table["experiment"]) == {
        "multimodal_fixed",
        "affect_only",
        "cspa_only",
        "cspa_affect",
    }

    effect_map = dict(zip(effects["effect"], effects["delta_mean"]))

    assert effect_map["affect_without_cspa"] == pytest.approx(-0.01)
    assert effect_map["cspa_without_affect"] == pytest.approx(-0.02)
    assert effect_map["affect_given_cspa"] == pytest.approx(0.05)
    assert effect_map["interaction"] == pytest.approx(0.06)
