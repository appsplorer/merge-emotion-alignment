#!/bin/bash
#SBATCH --job-name=merge_seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
SEEDS=(42 123 777 2026 3407)
CONFIGS=(multimodal_fixed cspa_affect)
S=${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}
C=${CONFIGS[$((SLURM_ARRAY_TASK_ID / 5))]}
python scripts/train.py --config configs/pipeline_experiments/${C}.yaml --set project.seed=$S
