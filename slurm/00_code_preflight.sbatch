#!/bin/bash
#SBATCH --job-name=merge_verify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
python - <<'PY'
from pathlib import Path
import optuna
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
print("GPU:", torch.cuda.get_device_name(0))
print("Optuna:", optuna.__version__)
models = [
    Path("/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache/m-a-p__MERT-v1-95M"),
    Path("/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache/FacebookAI__roberta-base"),
]
for path in models:
    assert path.is_dir(), path
    print("MODEL OK:", path)
PY
python -m compileall -q src scripts tests
python -m pytest tests -q
python scripts/build_manifest.py --profile 70-15-15
