from __future__ import annotations
import numpy as np


def paired_bootstrap_difference(y_true, pred_a, pred_b, metric, samples=5000, seed=2026):
    y=np.asarray(y_true); a=np.asarray(pred_a); b=np.asarray(pred_b); rng=np.random.default_rng(seed); n=len(y); diffs=[]
    for _ in range(samples):
        idx=rng.integers(0,n,n); diffs.append(metric(y[idx],b[idx])-metric(y[idx],a[idx]))
    return {"mean_difference":float(np.mean(diffs)),"ci_low":float(np.quantile(diffs,.025)),"ci_high":float(np.quantile(diffs,.975))}
