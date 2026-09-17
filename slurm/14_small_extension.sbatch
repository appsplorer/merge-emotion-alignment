#!/bin/bash
#SBATCH --job-name=merge_ext
#SBATCH --partition=gpu
#SBATCH --reservation=sa25abo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail

ROOT=/beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
ENV=/beegfs/general/sa25abo/research_workplace/environments/research_env
HF=/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache

cd "$ROOT"
source "$ENV/bin/activate"

export HF_HOME="$HF"
export HF_HUB_CACHE="$HF"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Prevent t-SNE / BLAS thread explosion.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

trap 'rc=$?; echo "[$(date -Iseconds)] FAILED rc=$rc line=$LINENO" >&2; exit $rc' ERR

echo "======================================================================"
echo "SMALL REVIEW-DEFENCE EXTENSION"
echo "======================================================================"

echo "started=$(date -Iseconds)"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "reservation=${SLURM_JOB_RESERVATION:-unknown}"
echo "expected_git_sha=${EXTENSION_BASE_SHA:-unset}"
echo "actual_git_sha=$(git rev-parse HEAD)"

if [ -z "${EXTENSION_BASE_SHA:-}" ]; then
    echo "ERROR: EXTENSION_BASE_SHA not supplied"
    exit 1
fi

if [ "$(git rev-parse HEAD)" != "$EXTENSION_BASE_SHA" ]; then
    echo "ERROR: repository SHA changed after submission"
    exit 1
fi

python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA required")

if torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one GPU")

print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())
PY

echo
echo "======================================================================"
echo "TESTS"
echo "======================================================================"

python -m pytest tests/test_small_extension.py -q

echo
echo "======================================================================"
echo "1/4 COLLISION AUDIT"
echo "======================================================================"

python scripts/audit_artist_title_collisions.py

echo
echo "======================================================================"
echo "2/4 EXPORT FROZEN VALIDATION REPRESENTATIONS"
echo "======================================================================"

python scripts/export_validation_embeddings.py \
  --experiment multimodal_noalign \
  --seed 42 \
  --require-cuda

python scripts/export_validation_embeddings.py \
  --experiment multimodal_fixed \
  --seed 42 \
  --require-cuda

python scripts/export_validation_embeddings.py \
  --experiment cspa_affect \
  --seed 42 \
  --require-cuda

echo
echo "======================================================================"
echo "3/4 THREE CSPA RAW-SGD-STYLE VALIDATION RUNS"
echo "======================================================================"

python scripts/run_cspa_sgd_robustness.py

echo
echo "======================================================================"
echo "4/4 ANALYSIS + FIGURES"
echo "======================================================================"

python scripts/analyze_small_extension.py

echo
echo "======================================================================"
echo "VERIFY NEW TRAINING TEST FIREWALL"
echo "======================================================================"

python - <<'PY'
import json
from pathlib import Path

for seed in (42, 123, 777):
    path = Path(
        f"results/runs/cspa_sgd_seed{seed}/metrics.json"
    )

    metrics = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert metrics["split"] == "validation"
    assert metrics["test_evaluated"] is False

    print(
        f"seed={seed}",
        f"macro_f1={metrics['macro_f1']:.6f}",
        "test_evaluated=False",
    )

print("NEW-TRAINING TEST FIREWALL: PASS")
PY

echo
echo "======================================================================"
echo "VERIFY ARTIFACTS"
echo "======================================================================"

python - <<'PY'
from pathlib import Path

required = [
    "results/tables/artist_title_collision_audit.csv",
    "results/tables/representation_projection_manifest.csv",

    "results/tables/cspa_optimizer_robustness.csv",
    "results/tables/cspa_optimizer_robustness_summary.csv",
    "results/tables/cspa_optimizer_robustness_paired.csv",

    "results/tables/collision_exclusion_test_sensitivity.csv",
    "results/tables/collision_exclusion_test_sensitivity_summary.csv",

    "results/manifests/artist_title_collision_resolution.json",
    "results/manifests/small_extension_manifest.json",

    "results/representations/multimodal_noalign_validation_seed42.npz",
    "results/representations/multimodal_noalign_validation_seed42.json",

    "results/representations/multimodal_fixed_validation_seed42.npz",
    "results/representations/multimodal_fixed_validation_seed42.json",

    "results/representations/cspa_affect_validation_seed42.npz",
    "results/representations/cspa_affect_validation_seed42.json",

    "figures/representation_tsne_modalities_validation.pdf",
    "figures/representation_tsne_modalities_validation.png",

    "figures/representation_tsne_fused_validation.pdf",
    "figures/representation_tsne_fused_validation.png",

    "figures/cspa_optimizer_robustness_validation.pdf",
    "figures/cspa_optimizer_robustness_validation.png",
]

failed = []

for name in required:
    path = Path(name)

    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        failed.append(name)
    else:
        print(
            "OK",
            name,
            path.stat().st_size,
            "bytes",
        )

if failed:
    raise SystemExit(
        "Missing/empty artifacts: "
        + repr(failed)
    )

print("ARTIFACT VERIFICATION: PASS")
PY

cat > results/manifests/small_extension_complete.txt <<EOF
completed=$(date -Iseconds)
job_id=${SLURM_JOB_ID:-unknown}
host=$(hostname)
reservation=${SLURM_JOB_RESERVATION:-unknown}
source_git_sha=${EXTENSION_BASE_SHA}
new_training_runs=3
new_training_seeds=42,123,777
new_training_split=validation
new_training_test_evaluated=false
representation_split=validation
representation_seed=42
EOF

echo
echo "======================================================================"
echo "FINAL RESULT PREVIEW"
echo "======================================================================"

echo
echo "--- collision audit ---"
cat results/tables/artist_title_collision_audit.csv

echo
echo "--- optimizer summary ---"
cat results/tables/cspa_optimizer_robustness_summary.csv

echo
echo "--- paired SGD minus AdamW ---"
cat results/tables/cspa_optimizer_robustness_paired.csv

echo
echo "--- completion record ---"
cat results/manifests/small_extension_complete.txt

echo
echo "======================================================================"
echo "GPU JOB COMPLETED SUCCESSFULLY"
echo "======================================================================"
