#!/bin/bash
#SBATCH --job-name=merge_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
python -m pytest tests -q
python scripts/train.py --config configs/pipeline_experiments/multimodal_noalign.yaml --set training.max_epochs=1 --set training.batch_size=8
