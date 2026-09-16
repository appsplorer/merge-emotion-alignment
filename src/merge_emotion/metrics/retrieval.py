from __future__ import annotations
import torch
import torch.nn.functional as F


def _direction(sim, ks):
    ranks=torch.argsort(sim,dim=1,descending=True); target=torch.arange(sim.size(0),device=sim.device)[:,None]; rank=(ranks==target).nonzero()[:,1]+1
    out={"r%d"%k:float((rank<=k).float().mean()) for k in ks}; out["median_rank"]=float(rank.float().median()); return out

def retrieval_metrics(a,l,ks=(1,5,10)):
    sim=F.normalize(a,dim=-1)@F.normalize(l,dim=-1).T; x=_direction(sim,ks); y=_direction(sim.T,ks); return {**{"retrieval_a2l_"+k:v for k,v in x.items()},**{"retrieval_l2a_"+k:v for k,v in y.items()}}
