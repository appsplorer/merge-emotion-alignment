#!/bin/bash
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
mkdir -p logs results/manifests

submit_after() {
  local dep="$1"; shift
  if [ -z "$dep" ]; then sbatch --parsable "$@"; else sbatch --parsable --dependency=afterok:${dep} "$@"; fi
}

J0=$(submit_after "" slurm/00_code_preflight.sbatch); echo "preflight       $J0"
J1=$(submit_after "$J0" slurm/01_audio_features.sbatch); echo "audio features  $J1"
J2=$(submit_after "$J1" slurm/02_lyrics_features.sbatch); echo "lyrics features $J2"
J3=$(submit_after "$J2" slurm/00_smoke.sbatch); echo "smoke           $J3"
J4=$(submit_after "$J3" slurm/03_tune.sbatch); echo "tuning          $J4"
J5=$(submit_after "$J4" slurm/04_baselines.sbatch); echo "baselines       $J5"
J6=$(submit_after "$J5" slurm/05_lambda_sweep.sbatch); echo "lambda sweep    $J6"
J7=$(submit_after "$J6" slurm/05b_select_lambda.sbatch); echo "select lambda   $J7"
J8=$(submit_after "$J7" slurm/06_optimizer_comparison.sbatch); echo "optimizers      $J8"
J9=$(submit_after "$J8" slurm/07_ablations.sbatch); echo "ablations       $J9"
J10=$(submit_after "$J9" slurm/08_final_seeds.sbatch); echo "final seeds     $J10"
J11=$(submit_after "$J10" slurm/09_select_finalists.sbatch); echo "freeze test set $J11"
J12=$(submit_after "$J11" slurm/10_final_evaluate.sbatch); echo "final test      $J12"
J13=$(submit_after "$J12" slurm/11_analysis.sbatch); echo "analysis        $J13"

echo "Final job: $J13"
echo "Monitor: squeue -u $USER"
