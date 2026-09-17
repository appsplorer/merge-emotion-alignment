from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


def normalize_lyrics_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    lines = [
        " ".join(line.split())
        for line in text.splitlines()
    ]
    return "\n".join(
        line
        for line in lines
        if line
    ).strip()


def normalize_identity_part(text: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        str(text),
    ).casefold()

    text = re.sub(
        r"[^\w]+",
        " ",
        text,
        flags=re.UNICODE,
    )

    return " ".join(
        text.split()
    ).strip()


def clean_identifier(value) -> str:
    if value is None:
        return ""

    try:
        if np.isnan(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if (
        text.endswith(".0")
        and text[:-2].isdigit()
    ):
        text = text[:-2]

    return text


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-12,
) -> float:
    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denom <= eps:
        return float("nan")

    return float(
        np.dot(x, y) / denom
    )


def classify_collision(
    audio_sha_equal: bool,
    lyrics_normalized_equal: bool,
    audio_cosine: float,
    lyrics_cosine: float,
    high_similarity_threshold: float = 0.9999,
) -> str:
    if (
        audio_sha_equal
        and lyrics_normalized_equal
    ):
        return "exact_duplicate_evidence"

    if (
        audio_sha_equal
        or lyrics_normalized_equal
    ):
        return "partial_duplicate_evidence"

    if (
        np.isfinite(audio_cosine)
        and np.isfinite(lyrics_cosine)
        and audio_cosine
        >= high_similarity_threshold
        and lyrics_cosine
        >= high_similarity_threshold
    ):
        return "high_representation_similarity"

    return "distinct_files_same_artist_title"


def build_theory_sgd_overrides(
    seed: int,
) -> list[str]:
    return [
        "project.seed=%d" % int(seed),
        "experiment.name=cspa_affect_sgd",
        "experiment.stage=optimizer_robustness",
        "optimizer.name=sgd",
        "optimizer.momentum=0.0",
        "optimizer.weight_decay=0.0",
        "optimizer.gradient_clip_norm=1000000000.0",
        "contrastive.lambda_align=0.2",
    ]


def pca_tsne_projection(
    matrix: np.ndarray,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    x = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if x.ndim != 2:
        raise ValueError(
            "Expected a 2-D embedding matrix"
        )

    if x.shape[0] < 4:
        raise ValueError(
            "Need at least four samples for t-SNE"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Embedding matrix contains non-finite values"
        )

    # L2 normalization is appropriate for representations used
    # with cosine-based cross-modal alignment.
    x = normalize(
        x,
        norm="l2",
        axis=1,
    )

    n_components = min(
        50,
        x.shape[1],
        x.shape[0] - 1,
    )

    if n_components >= 2:
        x = PCA(
            n_components=n_components,
            random_state=random_state,
        ).fit_transform(x)

    effective_perplexity = min(
        float(perplexity),
        max(
            2.0,
            (x.shape[0] - 1) / 3.0,
        ),
    )

    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(x)


def mean_cached_representation(
    payload,
    identifier: str,
) -> np.ndarray:
    ids = [
        str(value)
        for value
        in payload["ids"]
    ]

    try:
        index = ids.index(
            str(identifier)
        )
    except ValueError as exc:
        raise KeyError(
            "Identifier %s not in cache"
            % identifier
        ) from exc

    features = (
        payload["features"][index]
        .detach()
        .cpu()
        .float()
    )

    mask = payload.get("mask")

    if mask is None:
        pooled = features.mean(
            dim=0
        )
    else:
        current_mask = (
            mask[index]
            .detach()
            .cpu()
            .bool()
        )

        if current_mask.any():
            pooled = features[
                current_mask
            ].mean(
                dim=0
            )
        else:
            pooled = features.mean(
                dim=0
            )

    return pooled.numpy()
