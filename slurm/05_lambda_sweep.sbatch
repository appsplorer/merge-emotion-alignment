#!/bin/bash
#SBATCH --job-name=merge_lambda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-6%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
VALUES=(0.0 0.01 0.05 0.10 0.20 0.50 1.0)
L=${VALUES[$SLURM_ARRAY_TASK_ID]}
python scripts/train.py --config configs/pipeline_experiments/multimodal_fixed.yaml --set contrastive.lambda_align=$L --set experiment.name=lambda_${L}
