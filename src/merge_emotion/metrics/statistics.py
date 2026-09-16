from __future__ import annotations
import numpy as np


def bootstrap_ci(values, samples=5000, seed=2026):
    x=np.asarray(values,dtype=float); rng=np.random.default_rng(seed); means=[]
    for _ in range(samples): means.append(rng.choice(x,size=len(x),replace=True).mean())
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))
