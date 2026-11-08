from causal_suite.engine import attention_scale
from .common import *


def run(e,root,cfg):
 metas=measured(root);references={}
 for rel in sorted({m['relation'] for m in metas}):
  successes=[m for m in metas if m['split']=='calibration' and m['correct']]
  subjects={}
  for m in successes:
   if m['relation']==rel:subjects.setdefault(m['group_id'],[]).append(load_arrays(cache_path(root,m,m['template']))['head_margin'])
  if not subjects:continue
  avg=np.stack([np.stack(v).mean(0) for v in subjects.values()]).mean(0)
  flat=np.argsort(-avg.reshape(-1))[:cfg['sink_heads']];references[rel]=[(int(i//e.H),int(i%e.H)) for i in flat if avg.reshape(-1)[i]>0]
 dump(Path(root)/'sink_heads.json',references)
 for m in metas:
  if m['split']!='evaluation' or m['template']!=0:continue
  selected=references.get(m['relation'],[])
  if not selected:
   dump(Path(root)/'sink'/f'{stamp(m["id"],"coverage")}.json',dict(id=m['id'],status='no_positive_calibration_heads'));continue
  base=endpoint(logits(e,m),m)
  for kind in ['handoff','random']:
   for seed in (cfg['seeds'] if kind=='random' else [None]):
    if kind=='handoff':heads=selected
    else:
     indices=np.random.default_rng(seed).choice(e.L*e.H,len(selected),replace=False);heads=[(int(i//e.H),int(i%e.H)) for i in indices]
    for dose in cfg['sink_doses']:
     path=Path(root)/'sink'/f'{stamp(m["id"],kind,seed,dose)}.json'
     if path.exists():continue
     # If token zero is subject-origin, it is not a sink-only intervention.
     if 0 in m['source']:dump(path,dict(id=m['id'],status='first_token_is_subject'));continue
     with attention_scale(e,[0],heads,dose):ep=endpoint(logits(e,m,True),m)
     if dose==1 and abs(ep['margin']-base['margin'])>cfg['tolerance']*max(1,abs(base['margin'])):raise AssertionError('Sink identity intervention failed')
     dump(path,dict(id=m['id'],group_id=m['group_id'],relation=m['relation'],status='measured',head_kind=kind,seed=seed,dose=dose,heads=heads,baseline=base,endpoint=ep,margin_gain=ep['margin']-base['margin']))
