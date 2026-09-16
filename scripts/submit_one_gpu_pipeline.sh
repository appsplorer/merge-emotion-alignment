#!/bin/bash
set -Eeuo pipefail

ROOT=/beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
cd "$ROOT"
mkdir -p logs results/manifests

JOB_ID=$(sbatch --parsable slurm/12_full_pipeline_single_job.sbatch)

echo "full single-GPU job  $JOB_ID"
echo "allocation           one GPU"
echo "walltime             4-00:00:00"
echo "monitor              squeue -j $JOB_ID"
echo "stdout               logs/merge_full_${JOB_ID}.out"
echo "stderr               logs/merge_full_${JOB_ID}.err"
