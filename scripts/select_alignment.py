#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'results/runs').glob('lambda_*_seed42/metrics.json')):
    m=json.loads(p.read_text())
    run=p.parent
    cfg=(run/'config.yaml').read_text()
    import yaml
    c=yaml.safe_load(cfg)
    rows.append({'run_id':run.name,'lambda_align':float(c['contrastive']['lambda_align']),'macro_f1':float(m['macro_f1'])})
if not rows:
    raise SystemExit('No lambda sweep validation results found')
df=pd.DataFrame(rows).sort_values(['macro_f1','lambda_align'],ascending=[False,True])
best=df.iloc[0]
out=ROOT/'results/manifests'; out.mkdir(parents=True,exist_ok=True)
(out/'selected_lambda.txt').write_text(str(best['lambda_align'])+'\n')
(out/'selected_lambda.json').write_text(json.dumps(best.to_dict(),indent=2))
print('Selected lambda_align=',best['lambda_align'],'validation macro-F1=',best['macro_f1'])
