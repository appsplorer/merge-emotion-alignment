from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
from merge_emotion.metrics.classification import classification_metrics
from merge_emotion.metrics.alignment import cosine_alignment_metrics
from merge_emotion.metrics.retrieval import retrieval_metrics


def move_batch(batch, device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def evaluate(model, loader, device, modality):
    model.eval(); ys=[]; preds=[]; probs=[]; ids=[]; a_all=[]; l_all=[]
    with torch.inference_mode():
        for batch in loader:
            batch=move_batch(batch,device); out=model(batch,modality=modality); logits=out["multimodal_logits"] if modality=="multimodal" else out[modality+"_logits"]
            p=torch.softmax(logits,dim=-1); ys.extend(batch["label"].cpu().tolist()); preds.extend(p.argmax(-1).cpu().tolist()); probs.extend(p.cpu().tolist()); ids.extend(batch["song_id"])
            if modality=="multimodal": a_all.append(out["audio_repr"].cpu()); l_all.append(out["lyrics_repr"].cpu())
    metrics=classification_metrics(ys,preds)
    if a_all:
        a=torch.cat(a_all); l=torch.cat(l_all); metrics.update(cosine_alignment_metrics(a,l)); metrics.update(retrieval_metrics(a,l))
    rows=[{"song_id":sid,"y_true":int(y),"y_pred":int(pred),**{"p%d"%i:float(v) for i,v in enumerate(prob)}} for sid,y,pred,prob in zip(ids,ys,preds,probs)]
    return metrics, rows
