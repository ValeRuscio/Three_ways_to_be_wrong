"""Stage-1 OLMo2 trajectories with immutable revisions and fixed neuron IDs.
Final-checkpoint selection is retrospective; independent calibration sets and
same-fact final-success sets are explicitly separated.
"""
from pathlib import Path
import json,re,gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import HfApi
from causal_suite.data import digest
from causal_suite.engine import Engine
from .common import dump,key,cache_path,read_meta,load_arrays,table,evaluation
from .cache import capture_case,prepare,generate
from .factors import extract
from .selectors import build,select_site


def pin(cfg,path):
 path=Path(path)
 if path.exists():return json.loads(path.read_text())
 api=HfApi();repo=cfg['training_repo'];refs=api.list_repo_refs(repo);candidates={}
 for ref in list(refs.branches)+list(refs.tags):
  m=re.match(r'^stage1-step(\d+)-tokens',ref.name)
  if m:candidates[int(m.group(1))]=ref.name
 if cfg['training_revisions']:
  names=cfg['training_revisions']
  if any(not re.match(r'^stage1-step\d+-tokens',n) for n in names):raise ValueError('Training trajectory must stay within stage1; do not mix soup ingredients')
 else:
  if not candidates:raise ValueError('No official stage1 checkpoints found; provide explicit revisions')
  steps=sorted(candidates);names=list(dict.fromkeys(candidates[min(steps,key=lambda s:abs(s-requested))] for requested in cfg['training_steps']))
 result=[]
 for name in names:
  step=int(re.search(r'stage1-step(\d+)',name).group(1));sha=api.model_info(repo,revision=name).sha
  result.append(dict(repo=repo,revision=name,sha=sha,step=step))
 result.sort(key=lambda s:s['step'])
 if len(result)<2:raise ValueError('Need at least two distinct training checkpoints')
 dump(path,result);return result


def load(spec,cfg):
 tok=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['sha'])
 model=AutoModelForCausalLM.from_pretrained(spec['repo'],revision=spec['sha'],torch_dtype=torch.float32,attn_implementation='eager').to(cfg['device']);return tok,Engine(model)


def fixed_selection(records,final_root,index,cfg):
 fixed={}
 for r in records:
  m=read_meta(final_root,r,0)
  if m.get('status')!='measured':continue
  entry=index.get(r['relation'],{})
  if entry.get('status')=='ready':ref=load_arrays(Path(final_root)/'selectors'/entry['file']);kind='independent_final_calibration'
  elif m['correct']:ref=load_arrays(cache_path(final_root,r,0));kind='retrospective_final_fact'
  else:fixed[str(r['id'])]=dict(kind='no_reference',sites=[]);continue
  sites=[select_site(ref,s,cfg['top_k']) for s in cfg['sites']]
  fixed[str(r['id'])]=dict(kind=kind,sites=[s for s in sites if s is not None])
 return fixed


def rows_from(a,selection):
 rows=[]
 for item in selection['sites']:
  l,s=item['layer'],item['site_index']
  for n in item['neurons']:
   rows.append(dict(set_kind=selection['kind'],site=item['site'],layer=l,neuron=n,
    **{k:float(a[k][l,s,n]) for k in ['pre','gate','sigmoid','up_subject','up_cosine','down_gold','down_cosine','contribution_gold','signed_distance']}))
 return rows


def run(records,out,cfg):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);specs=pin(cfg,out/'checkpoints.json');ev=evaluation(records,cfg);cal=[r for r in records if r.get('eligible') and r['split']=='calibration'];final=specs[-1];fr=out/'final_reference'
 final_done=out/'fixed_neurons.json'
 if not final_done.exists():
  tok,e=load(final,cfg);local=dict(cfg,save_frozen_cache=False)
  for i,r in enumerate(cal+ev):
   capture_case(e,tok,r,0,fr,local,calibration=r['split']=='calibration')
   if i%25==0:print(f'OLMo final-reference: {i+1}/{len(cal)+len(ev)}',flush=True)
  index=build(cal,fr,cfg);fixed=fixed_selection(ev,fr,index,cfg);dump(final_done,fixed)
  dump(out/'tokenizer.json',dict(hash=digest(sorted(tok.get_vocab().items()))))
  del e,tok;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 fixed=json.loads(final_done.read_text());token_hash=json.loads((out/'tokenizer.json').read_text())['hash']
 for spec in specs:
  folder=out/f'step_{spec["step"]}';folder.mkdir(exist_ok=True)
  todo=[r for r in ev if not (folder/f'{key(r)}.json').exists()]
  if not todo:continue
  # Final reference features can be reused without reloading final weights.
  if spec==final:
   for r in todo:
    m=read_meta(fr,r,0);selection=fixed.get(str(r['id']),dict(kind='no_reference',sites=[]));rows=rows_from(load_arrays(cache_path(fr,r,0)),selection) if m.get('status')=='measured' else []
    dump(folder/f'{key(r)}.json',dict(id=str(r['id']),group_id=str(r['group_id']),relation=r['relation'],step=spec['step'],sha=spec['sha'],meta=m,rows=rows))
   continue
  tok,e=load(spec,cfg)
  if digest(sorted(tok.get_vocab().items()))!=token_hash:raise ValueError('Tokenizer vocabulary changed across checkpoints')
  for i,r in enumerate(todo):
   try:m=prepare(tok,r,0)
   except (ValueError,KeyError,TypeError) as ex:
    dump(folder/f'{key(r)}.json',dict(id=str(r['id']),group_id=str(r['group_id']),relation=r['relation'],step=spec['step'],sha=spec['sha'],meta=dict(status='excluded',reason=str(ex)),rows=[]));continue
   C=e.capture(torch.tensor(m['input_ids'],device=e.device),tol=cfg['tolerance']);g=m['g'];other=C['z'].clone();other[g]=-torch.inf;c=int(other.argmax());top=int(C['z'].argmax())==g;text=None;correct=False
   if top:text,correct=generate(e,tok,C['ids'],m['aliases'],cfg)
   m.update(c=c,top1=top,correct=bool(top and correct),generated=text,margin=float(C['z'][g]-C['z'][c]),status='measured',certificate=C['certificate'])
   selection=fixed.get(str(r['id']),dict(kind='no_reference',sites=[]));rows=[]
   if selection['sites']:rows=rows_from(extract(e,C,m['source'],g,c,cfg['tolerance']),selection)
   dump(folder/f'{key(r)}.json',dict(id=str(r['id']),group_id=str(r['group_id']),relation=r['relation'],step=spec['step'],sha=spec['sha'],meta=m,rows=rows))
   if i%10==0:print(f'OLMo step {spec["step"]}: {i+1}/{len(todo)}',flush=True)
  del e,tok;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 summarize(out,cfg)


def first_sustained(values,n):
 for i in range(len(values)-n+1):
  if all(values[i:i+n]):return i
 return None


def summarize(out,cfg):
 out=Path(out);states=[];rows=[]
 for p in sorted(out.glob('step_*/*.json')):
  d=json.loads(p.read_text());m=d['meta'];base={k:d[k] for k in ['id','group_id','relation','step','sha']}
  states.append(dict(**base,status=m['status'],correct=m.get('correct'),gold_top1=m.get('top1'),margin=m.get('margin')))
  rows.extend(dict(**base,**r) for r in d['rows'])
 table(states,out/'reports/training_status.csv');d=table(rows,out/'reports/training_neurons.csv');events=[];progress=[];aligned=[]
 if not len(d):dump(out/'reports/training_status.json',dict(status='no_supported_neuron_sets'));return
 allsteps=[s['step'] for s in json.loads((out/'checkpoints.json').read_text())]
 st=pd.DataFrame(states)
 for id,s in st.groupby('id'):
  s=s.sort_values('step')
  if list(s.step)!=allsteps or not all(s.status=='measured'):continue
  correct=s.correct.astype(bool).tolist();event=first_sustained(correct,cfg['training_sustained'])
  # Initial successes are left-censored, never called newly learned facts.
  newly=event is not None and event>0 and not correct[0]
  events.append(dict(id=id,group_id=s.iloc[0].group_id,relation=s.iloc[0].relation,newly_correct=newly,first_sustained_index=event,
   first_sustained_step=allsteps[event] if event is not None else None,previous_observed_step=allsteps[event-1] if event is not None and event>0 else None,initial_correct=correct[0],final_correct=correct[-1]))
  if not newly:continue
  case=d[d.id==id]
  for (kind,site),part in case.groupby(['set_kind','site']):
   avg=part.groupby('step')[['pre','up_cosine','down_cosine']].mean().reindex(allsteps)
   for j,step in enumerate(allsteps):aligned.append(dict(id=id,group_id=s.iloc[0].group_id,set_kind=kind,site=site,relative_checkpoint=j-event,step=step,**avg.loc[step].to_dict()))
   for metric in ['pre','up_cosine','down_cosine']:
    v=avg[metric].to_numpy();span=v[-1]-v[0]
    if not np.isfinite(v).all() or abs(span)<1e-12:continue
    p=(v-v[0])/span
    for level in cfg['training_progress_levels']:
     crossing=first_sustained((p>=level).tolist(),cfg['training_sustained'])
     progress.append(dict(id=id,set_kind=kind,site=site,metric=metric,progress_level=level,initial=v[0],final=v[-1],net_increase=span>0,
      crossing_step=allsteps[crossing] if crossing is not None else None,interpretation='Progress along observed initial-to-final change, not a physical gate-opening or acquisition threshold'))
 table(events,out/'reports/training_events.csv');table(aligned,out/'reports/training_event_aligned.csv');table(progress,out/'reports/training_progress.csv')
 dump(out/'reports/training_interpretation.json',dict(neuron_ids='Fixed from final stage1 checkpoint; independent calibration and retrospective per-fact sets remain separated',
  steps='Irregular sampled checkpoints; learning time is interval-censored. First sustained canonical+alias success, not proof of training exposure.',
  ordering='Preactivation and live-pass downstream alignment both depend on activations and weights. Trajectory ordering does not isolate which parameter family learned first.',
  gradients='Unavailable unless actual training gradient snapshots and a compatible independently estimated basis are supplied.'))
