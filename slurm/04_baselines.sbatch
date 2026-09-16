#!/bin/bash
#SBATCH --job-name=merge_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
CONFIGS=(audio_only lyrics_only multimodal_noalign multimodal_fixed)
C=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
python scripts/train.py --config configs/pipeline_experiments/${C}.yaml --tuned results/manifests/optuna_best.yaml --run-id baseline_${C}_seed42 --resume auto --require-cuda --set experiment.stage=baseline
