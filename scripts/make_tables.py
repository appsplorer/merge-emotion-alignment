#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/"results/summaries/all_runs.csv"; df=pd.read_csv(src); out=ROOT/"results/tables"; out.mkdir(parents=True,exist_ok=True)
if not df.empty:
    main=df.groupby("experiment",as_index=False).agg(macro_f1_mean=("macro_f1","mean"),macro_f1_sd=("macro_f1","std"),accuracy_mean=("accuracy","mean"),n=("seed","count")); main.to_csv(out/"main_results.csv",index=False)
    if "lambda_align" in df.columns: df[["experiment","seed","lambda_align","macro_f1","alignment_gap","retrieval_a2l_r5"]].dropna(subset=["lambda_align"]).to_csv(out/"lambda_sensitivity.csv",index=False)
print("Saved tables to",out)
