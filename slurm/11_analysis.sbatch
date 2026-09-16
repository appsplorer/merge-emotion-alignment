#!/bin/bash
#SBATCH --job-name=merge_analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment
source /beegfs/general/sa25abo/research_workplace/environments/research_env/bin/activate
python scripts/aggregate_results.py
python scripts/make_tables.py
python scripts/analyze_final_statistics.py
python scripts/make_figures.py
