#!/usr/bin/env python3
from pathlib import Path
from merge_emotion.analysis.aggregate import collect_runs, summarize_runs
ROOT=Path(__file__).resolve().parents[1]
runs=collect_runs(ROOT/"results/runs"); summary=summarize_runs(runs); out=ROOT/"results/summaries/all_runs.csv"; out.parent.mkdir(parents=True,exist_ok=True); summary.to_csv(out,index=False); print("Saved",out)
