#!/bin/bash
#SBATCH --job-name=merge_ablate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=26G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
CONFIGS=(multimodal_fixed affect_only cspa_only cspa_affect)
C=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
L=$(cat results/manifests/selected_lambda.txt)
python scripts/train.py --config configs/pipeline_experiments/${C}.yaml --tuned results/manifests/optuna_best.yaml --run-id ablation_${C}_seed42 --resume auto --require-cuda --set contrastive.lambda_align=$L --set experiment.stage=ablation
