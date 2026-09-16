#!/bin/bash
#SBATCH --job-name=merge_ablate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
CONFIGS=(multimodal_fixed affect_only cspa_only cspa_affect)
python scripts/train.py --config configs/pipeline_experiments/${CONFIGS[$SLURM_ARRAY_TASK_ID]}.yaml
