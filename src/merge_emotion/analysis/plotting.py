from pathlib import Path
import matplotlib.pyplot as plt


def save_main_performance(df,path:Path):
    means=df.groupby("experiment")["macro_f1"].mean().sort_values(); ax=means.plot(kind="barh",figsize=(7,4)); ax.set_xlabel("Macro-F1"); ax.set_ylabel(""); ax.figure.tight_layout(); ax.figure.savefig(path,bbox_inches="tight"); ax.figure.savefig(path.with_suffix(".png"),dpi=220,bbox_inches="tight"); plt.close(ax.figure)

def save_alignment_scatter(df,path:Path):
    data=df.dropna(subset=["alignment_gap","macro_f1"]); fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(data["alignment_gap"],data["macro_f1"]); ax.set_xlabel("Matched-minus-unmatched cosine similarity"); ax.set_ylabel("Macro-F1"); fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); fig.savefig(path.with_suffix(".png"),dpi=220,bbox_inches="tight"); plt.close(fig)

def save_pareto(df,path:Path):
    save_alignment_scatter(df,path)
