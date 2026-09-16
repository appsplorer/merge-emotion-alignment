from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def collect_runs(run_root: Path):
    rows=[]
    for p in sorted(run_root.glob("*")):
        if not (p/"metrics.json").exists(): continue
        row=json.loads((p/"metrics.json").read_text()); row["run_dir"]=str(p)
        cfg=p/"config.yaml"
        if cfg.exists():
            import yaml; c=yaml.safe_load(cfg.read_text()); row["lambda_align"]=c.get("contrastive",{}).get("lambda_align"); row["strategy"]=c.get("gradient_strategy",{}).get("name"); row["affect_aware"]=c.get("affect_aware",{}).get("enabled",False); row["cspa"]=c.get("cspa",{}).get("enabled",False)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_runs(df):
    return df.copy() if df.empty else df.sort_values(["experiment","seed"]).reset_index(drop=True)
