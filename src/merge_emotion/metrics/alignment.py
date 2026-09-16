from __future__ import annotations
import torch
import torch.nn.functional as F


def cosine_alignment_metrics(audio_repr, lyrics_repr):
    a=F.normalize(audio_repr,dim=-1); l=F.normalize(lyrics_repr,dim=-1); sim=a@l.T
    n=sim.size(0); eye=torch.eye(n,dtype=torch.bool,device=sim.device); pos=sim.diag().mean(); neg=sim.masked_select(~eye).mean()
    return {"alignment_positive_cosine":float(pos),"alignment_negative_cosine":float(neg),"alignment_gap":float(pos-neg)}
