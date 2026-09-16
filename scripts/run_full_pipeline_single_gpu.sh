#!/bin/bash
set -Eeuo pipefail

ROOT=/beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
ENV=/beegfs/general/sa25abo/research_workplace/environments/research_env
STATE_DIR="$ROOT/results/manifests/single_gpu_state"

cd "$ROOT"
source "$ENV/bin/activate"
mkdir -p logs results/manifests "$STATE_DIR"

export HF_HOME=/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

trap 'rc=$?; echo "[$(date -Is)] FAILED rc=${rc} line=${LINENO}" >&2; exit "$rc"' ERR

run_once() {
  local key="$1"; shift
  local label="$1"; shift
  local marker="$STATE_DIR/${key}.done"
  if [[ -f "$marker" ]]; then
    echo "[$(date -Is)] SKIP: $label ($marker exists)"
    return 0
  fi
  echo
  echo "======================================================================"
  echo "[$(date -Is)] START: $label"
  echo "======================================================================"
  "$@"
  printf '%s\n' "completed=$(date -Is)" "job_id=${SLURM_JOB_ID:-none}" "git_sha=$(git rev-parse HEAD)" > "$marker"
  echo "[$(date -Is)] DONE: $label"
}

preflight() {
  python - <<'PY'
from pathlib import Path
import optuna
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, f"Expected exactly one visible GPU, got {torch.cuda.device_count()}"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
print("GPU:", torch.cuda.get_device_name(0))
print("Optuna:", optuna.__version__)
for path in [
    Path("/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache/m-a-p__MERT-v1-95M"),
    Path("/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache/FacebookAI__roberta-base"),
]:
    assert path.is_dir(), path
    print("MODEL OK:", path)
PY
  python -m compileall -q src scripts tests
  python -m pytest tests -q
  python scripts/build_manifest.py --profile 70-15-15
}

extract_audio() {
  python scripts/extract_audio_features.py --require-cuda
}

extract_lyrics() {
  python scripts/extract_lyrics_features.py --require-cuda
}

smoke() {
  python scripts/train.py \
    --config configs/pipeline_experiments/multimodal_noalign.yaml \
    --run-id smoke_multimodal_noalign_seed42 \
    --resume auto --require-cuda \
    --set training.max_epochs=1 \
    --set training.batch_size=8 \
    --set experiment.stage=smoke
}

tune() {
  python scripts/tune.py --trials 30 --require-cuda
}

run_train() {
  python scripts/train.py "$@"
}

freeze_finalists() {
  python scripts/aggregate_results.py
  python scripts/select_finalists.py
}

final_test() {
  python scripts/final_evaluate.py \
    --selection-manifest results/manifests/final_selection.yaml \
    --require-cuda
}

analysis() {
  python scripts/aggregate_results.py
  python scripts/make_tables.py
  python scripts/analyze_final_statistics.py
  python scripts/make_figures.py
}

JOB_RECORD="results/manifests/single_gpu_job_${SLURM_JOB_ID:-manual}.txt"
printf '%s\n' \
  "started=$(date -Is)" \
  "job_id=${SLURM_JOB_ID:-none}" \
  "host=$(hostname)" \
  "git_sha=$(git rev-parse HEAD)" \
  "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}" \
  > "$JOB_RECORD"

run_once preflight "code/data/GPU preflight" preflight
run_once audio_features "MERT feature extraction" extract_audio
run_once lyrics_features "RoBERTa feature extraction" extract_lyrics
run_once smoke "one-epoch real-data smoke training" smoke
run_once tuning "30-trial Optuna tuning" tune

for C in audio_only lyrics_only multimodal_noalign multimodal_fixed; do
  run_once "baseline_${C}_seed42" "baseline: ${C}, seed 42" \
    run_train \
      --config "configs/pipeline_experiments/${C}.yaml" \
      --tuned results/manifests/optuna_best.yaml \
      --run-id "baseline_${C}_seed42" \
      --resume auto --require-cuda \
      --set experiment.stage=baseline
done

for L in 0.0 0.01 0.05 0.10 0.20 0.50 1.0; do
  run_once "lambda_${L}_seed42" "lambda sweep: ${L}" \
    run_train \
      --config configs/pipeline_experiments/multimodal_fixed.yaml \
      --tuned results/manifests/optuna_best.yaml \
      --run-id "lambda_${L}_seed42" \
      --resume auto --require-cuda \
      --set "contrastive.lambda_align=${L}" \
      --set "experiment.name=lambda_${L}" \
      --set experiment.stage=lambda_sweep
done

run_once select_lambda "freeze validation-selected alignment weight" python scripts/select_alignment.py
SELECTED_LAMBDA=$(cat results/manifests/selected_lambda.txt)
echo "Selected lambda: $SELECTED_LAMBDA"

for C in multimodal_fixed multimodal_pcgrad multimodal_cagrad cspa_affect; do
  run_once "optimizer_${C}_seed42" "optimizer comparison: ${C}, seed 42" \
    run_train \
      --config "configs/pipeline_experiments/${C}.yaml" \
      --tuned results/manifests/optuna_best.yaml \
      --run-id "optimizer_${C}_seed42" \
      --resume auto --require-cuda \
      --set "contrastive.lambda_align=${SELECTED_LAMBDA}" \
      --set experiment.stage=optimizer_comparison
done

for C in multimodal_fixed affect_only cspa_only cspa_affect; do
  run_once "ablation_${C}_seed42" "ablation: ${C}, seed 42" \
    run_train \
      --config "configs/pipeline_experiments/${C}.yaml" \
      --tuned results/manifests/optuna_best.yaml \
      --run-id "ablation_${C}_seed42" \
      --resume auto --require-cuda \
      --set "contrastive.lambda_align=${SELECTED_LAMBDA}" \
      --set experiment.stage=ablation
done

METHODS=(audio_only lyrics_only multimodal_noalign multimodal_fixed multimodal_pcgrad multimodal_cagrad affect_only cspa_only cspa_affect)
SEEDS=(42 123 777 2026 3407)
for M in "${METHODS[@]}"; do
  for S in "${SEEDS[@]}"; do
    run_once "final_${M}_seed${S}" "final validation run: ${M}, seed ${S}" \
      run_train \
        --config "configs/pipeline_experiments/${M}.yaml" \
        --tuned results/manifests/optuna_best.yaml \
        --run-id "final_${M}_seed${S}" \
        --resume auto --require-cuda \
        --set "project.seed=${S}" \
        --set "contrastive.lambda_align=${SELECTED_LAMBDA}" \
        --set experiment.stage=final_seed
  done
done

run_once freeze_finalists "aggregate validation results and freeze final checkpoint hashes" freeze_finalists
run_once final_test "held-out test evaluation of frozen checkpoints" final_test
run_once final_analysis "bootstrap statistics, tables, and figures" analysis

printf '%s\n' "completed=$(date -Is)" >> "$JOB_RECORD"
echo
echo "======================================================================"
echo "FULL SINGLE-GPU PIPELINE COMPLETED"
echo "Job record: $JOB_RECORD"
echo "======================================================================"
