from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred):
    p,r,f,s=precision_recall_fscore_support(y_true,y_pred,labels=[0,1,2,3],zero_division=0)
    out={"accuracy":float(accuracy_score(y_true,y_pred)),"macro_f1":float(f1_score(y_true,y_pred,average="macro")),"weighted_f1":float(f1_score(y_true,y_pred,average="weighted"))}
    for i in range(4): out.update({"q%d_precision"%(i+1):float(p[i]),"q%d_recall"%(i+1):float(r[i]),"q%d_f1"%(i+1):float(f[i]),"q%d_support"%(i+1):int(s[i])})
    return out
