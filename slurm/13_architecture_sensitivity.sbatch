#!/bin/bash
#SBATCH --job-name=merge_archsens
#SBATCH --partition=gpu
#SBATCH --reservation=sa25abo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

trap 'rc=$?; echo "[$(date -Iseconds)] FAILED rc=$rc line=$LINENO" >&2; exit $rc' ERR

echo "======================================================================"
echo "ARCHITECTURE SENSITIVITY JOB"
echo "======================================================================"
echo "started=$(date -Iseconds)"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "git_sha=$(git rev-parse HEAD)"
echo "reservation=${SLURM_JOB_RESERVATION:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
if torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one visible GPU")
print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())
PY

echo
echo "======================================================================"
echo "PROTOCOL TEST"
echo "======================================================================"

python -m pytest tests/test_architecture_sensitivity.py -q

echo
echo "======================================================================"
echo "RUNNING 9 ARCHITECTURES x 3 SEEDS = 27 VALIDATION RUNS"
echo "NO HELD-OUT TEST EVALUATION"
echo "======================================================================"

python scripts/run_architecture_sensitivity.py

echo
echo "======================================================================"
echo "ANALYSIS + ABLATION + FIGURES"
echo "======================================================================"

python scripts/analyze_architecture_sensitivity.py

echo
echo "======================================================================"
echo "ARTIFACT VERIFICATION"
echo "======================================================================"

python - <<'PY'
from pathlib import Path
import json
import pandas as pd

required = [
    Path("results/tables/architecture_sensitivity_validation.csv"),
    Path("results/tables/architecture_sensitivity_summary.csv"),
    Path("results/tables/architecture_sensitivity_paired_vs_base.csv"),
    Path("results/tables/component_ablation_final_test.csv"),
    Path("results/tables/component_ablation_paired_effects.csv"),
    Path("figures/architecture_sensitivity_validation.pdf"),
    Path("figures/architecture_sensitivity_validation.png"),
    Path("figures/component_ablation_final_test.pdf"),
    Path("figures/component_ablation_final_test.png"),
    Path("results/manifests/architecture_sensitivity_manifest.json"),
]

for path in required:
    if not path.exists():
        raise SystemExit("MISSING: %s" % path)
    if path.stat().st_size == 0:
        raise SystemExit("EMPTY: %s" % path)
    print("OK:", path, path.stat().st_size, "bytes")

raw = pd.read_csv(
    "results/tables/architecture_sensitivity_validation.csv"
)

if len(raw) != 27:
    raise SystemExit(
        "Expected 27 architecture runs, found %d" % len(raw)
    )

if set(raw["seed"]) != {42, 123, 777}:
    raise SystemExit("Unexpected seed set")

if not (raw["split"] == "validation").all():
    raise SystemExit("Non-validation architecture record detected")

if raw["test_evaluated"].astype(str).str.lower().isin(
    ["true", "1"]
).any():
    raise SystemExit("Test firewall violation detected")

manifest = json.loads(
    Path(
        "results/manifests/architecture_sensitivity_manifest.json"
    ).read_text(encoding="utf-8")
)

print()
print("Architecture rows:", len(raw))
print("Manifest git SHA:", manifest["git_sha"])
print("TEST FIREWALL: PASS")
print("ARTIFACT VERIFICATION: PASS")
PY

cat > results/manifests/architecture_sensitivity_complete.txt <<EOF
completed=$(date -Iseconds)
job_id=${SLURM_JOB_ID:-unknown}
host=$(hostname)
git_sha=$(git rev-parse HEAD)
reservation=${SLURM_JOB_RESERVATION:-unknown}
architecture_runs=27
architecture_split=validation
architecture_test_evaluated=false
EOF

echo
echo "======================================================================"
echo "ARCHITECTURE SENSITIVITY COMPLETED SUCCESSFULLY"
echo "======================================================================"
cat results/manifests/architecture_sensitivity_complete.txt
