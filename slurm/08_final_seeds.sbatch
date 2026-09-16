#!/bin/bash
#SBATCH --job-name=merge_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-44%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
METHODS=(audio_only lyrics_only multimodal_noalign multimodal_fixed multimodal_pcgrad multimodal_cagrad affect_only cspa_only cspa_affect)
SEEDS=(42 123 777 2026 3407)
M_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
S_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
M=${METHODS[$M_INDEX]}
S=${SEEDS[$S_INDEX]}
L=$(cat results/manifests/selected_lambda.txt)
python scripts/train.py --config configs/pipeline_experiments/${M}.yaml --tuned results/manifests/optuna_best.yaml --run-id final_${M}_seed${S} --resume auto --require-cuda --set project.seed=$S --set contrastive.lambda_align=$L --set experiment.stage=final_seed
