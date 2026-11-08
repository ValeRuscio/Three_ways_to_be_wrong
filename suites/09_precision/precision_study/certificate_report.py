import argparse
from pathlib import Path
import pandas as pd
from channel_study.io import dump,load


def report(root):
 root=Path(root);rows=[];identities=[];baselines=[]
 for path in root.glob('*/prompts/*/certificate.json'):
  d=load(path);baseline=load(path.parent/'baseline.json');p=baseline['prompt']
  meta=dict(model=path.parents[2].name,id=p['id'],group_id=p['group_id'],template=p['template'])
  rows.extend(dict(meta,**x) for x in d['rows']);identities.extend(dict(meta,**x) for x in d.get('split_identity',[]))
  baselines.append(dict(meta,baseline_disagreement=d['baseline_disagreement'],native_certificate=d['native_certificate']))
 out=root/'reports';out.mkdir(exist_ok=True)
 if not rows:return
 df=pd.DataFrame(rows);df.to_csv(out/'certificate_per_check.csv',index=False)
 pd.DataFrame(identities).to_csv(out/'split_identity_per_matrix.csv',index=False)
 pd.DataFrame(baselines).to_csv(out/'certificate_baselines.csv',index=False)
 groups=['model','layer','family','direction','epsilon']
 df.groupby(groups).agg(checks=('live_pass','size'),live_failures=('live_pass',lambda x:int((~x).sum())),frozen_failures=('frozen_pass',lambda x:int((~x).sum())),live_unresolved=('live_resolved',lambda x:int((~x).sum())),frozen_unresolved=('frozen_resolved',lambda x:int((~x).sum())),max_live_error=('live_absolute_error','max'),max_frozen_error=('frozen_absolute_error','max')).reset_index().to_csv(out/'certificate_failure_table.csv',index=False)
 dump(out/'certificate_audit.json',dict(checks=len(df),live_failures=int((~df.live_pass).sum()),frozen_failures=int((~df.frozen_pass).sum()),live_unresolved=int((~df.live_resolved).sum()),frozen_unresolved=int((~df.frozen_resolved).sum()),
  max_split_closure=max(x['closure_error'] for x in identities) if identities else None,
  interpretation='Sampled numerical validation, NOT a proof for all weights, prompts or quantities. Unresolved near-zero slopes are not informative successes. Frozen q/k/gate gradients are structurally zero; finite edit tests should preserve the frozen objective. Native finite differences have truncation and rounding error; inspect epsilon trends and failures.'))
 print('Certificate audit:',load(out/'certificate_audit.json'),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();report(a.root)
if __name__=='__main__':main()
