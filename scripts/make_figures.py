#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from merge_emotion.analysis.plotting import save_alignment_scatter, save_main_performance, save_pareto
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"results/summaries/all_runs.csv"); out=ROOT/"figures"; out.mkdir(parents=True,exist_ok=True)
if not df.empty:
    save_main_performance(df,out/"main_performance.pdf"); save_alignment_scatter(df,out/"alignment_vs_f1.pdf"); save_pareto(df,out/"pareto_frontier.pdf")
print("Saved figures to",out)
