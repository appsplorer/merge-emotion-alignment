#!/bin/bash
#SBATCH --job-name=merge_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
python scripts/train.py --config configs/pipeline_experiments/multimodal_noalign.yaml --run-id smoke_multimodal_noalign_seed42 --resume auto --require-cuda --set training.max_epochs=1 --set training.batch_size=8 --set experiment.stage=smoke
