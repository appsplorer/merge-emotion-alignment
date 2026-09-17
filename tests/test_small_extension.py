from __future__ import annotations

import numpy as np
import pytest


def _feature():
    try:
        from merge_emotion.analysis.small_extension import (
            build_theory_sgd_overrides,
            classify_collision,
            normalize_lyrics_text,
            pca_tsne_projection,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            "Small-extension feature does not exist yet: %s" % exc
        )

    return (
        build_theory_sgd_overrides,
        classify_collision,
        normalize_lyrics_text,
        pca_tsne_projection,
    )


def test_normalized_lyrics_hash_input_is_stable():
    _, _, normalize, _ = _feature()

    a = " Hello   WORLD!\nThis is\tA Song. "
    b = "hello world!\nthis is a song."

    assert normalize(a) == normalize(b)


def test_collision_classification_prefers_exact_evidence():
    _, classify, _, _ = _feature()

    assert (
        classify(
            audio_sha_equal=True,
            lyrics_normalized_equal=True,
            audio_cosine=1.0,
            lyrics_cosine=1.0,
        )
        == "exact_duplicate_evidence"
    )

    assert (
        classify(
            audio_sha_equal=False,
            lyrics_normalized_equal=False,
            audio_cosine=0.45,
            lyrics_cosine=0.51,
        )
        == "distinct_files_same_artist_title"
    )


def test_sgd_overrides_match_raw_sgd_diagnostic():
    build, _, _, _ = _feature()

    overrides = build(42)

    assert "project.seed=42" in overrides
    assert "experiment.stage=optimizer_robustness" in overrides
    assert "optimizer.name=sgd" in overrides
    assert "optimizer.momentum=0.0" in overrides
    assert "optimizer.weight_decay=0.0" in overrides

    # Effectively disable gradient clipping to preserve the raw
    # SGD direction assumed by the CSPA descent calculation.
    assert "optimizer.gradient_clip_norm=1000000000.0" in overrides


def test_projection_is_deterministic_and_two_dimensional():
    _, _, _, project = _feature()

    rng = np.random.RandomState(7)
    x = rng.normal(size=(12, 4)).astype("float32")

    a = project(
        x,
        random_state=42,
        perplexity=2,
    )

    b = project(
        x,
        random_state=42,
        perplexity=2,
    )

    assert a.shape == (12, 2)
    assert b.shape == (12, 2)
    assert np.allclose(a, b)
