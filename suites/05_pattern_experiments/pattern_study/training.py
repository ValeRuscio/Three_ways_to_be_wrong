"""OLMo-native, final-success retrospective cohorts, independent subject splits."""
import re
from pathlib import Path
from .common import *
from .capture import screen,run as capture
from gate_study.variance import run as variance


def checkpoints(spec,cfg,path):
 if Path(path).exists():return load(path)
 from huggingface_hub import HfApi
 api=HfApi();refs=api.list_repo_refs(spec['repo']);available={}
 for r in list(refs.branches)+list(refs.tags):
  m=re.match(r'^stage1-step(\d+)-tokens',r.name)
  if m:available[int(m[1])]=r.name
 if len(available)<2:raise ValueError('Need at least two official stage1 checkpoints for '+spec['repo'])
 final=max(available);steps=sorted({min(available,key=lambda s:abs(s-final*f)) for f in cfg['training_fractions']}|{final})
 result=[dict(repo=spec['repo'],step=s,revision=available[s],sha=api.model_info(spec['repo'],revision=available[s]).sha) for s in steps]
 dump(path,result);return result


def broad_cohort(cfg,out):
 from causal_suite.data import build_fixed
 c=dict(seed=cfg['seed'],facts_jsonl=cfg['training_facts_jsonl'],calibration_candidates_per_relation=10**9,evaluation_per_relation=10**9)
 raw=build_fixed(c,out)
 return [r for r in raw['rows'] if r['split'] in ['calibration','evaluation'] and isinstance(r.get('subject'),str) and r['subject'].strip()]


def first_sustained(values,n):
 for i in range(len(values)-n+1):
  if all(values[i:i+n]):return i
 return None


def run(spec,root,cfg):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);specs=checkpoints(spec,cfg,root/'checkpoints.json');pool=broad_cohort(cfg,root/'candidate_data');cal=[r for r in pool if r['split']=='calibration']
 final=specs[-1];selection=root/'selected_facts.json'
 if not selection.exists():
  tok,e=model_load(final,cfg);selected=[];counts={}
  for i,r in enumerate(pool):
   path=root/'screen'/f'{key(r)}.json'
   m=load(path) if path.exists() else screen(e,tok,r,0,cal,cfg)
   if not path.exists():dump(path,m)
   counts[r['split']]=counts.get(r['split'],0)+1
   if m.get('correct'):selected.append(r)
   if i%25==0:print(f'{spec["key"]} final checkpoint screening {i+1}/{len(pool)}',flush=True)
  dump(selection,dict(records=selected,screened=counts,selected={s:sum(r['split']==s for r in selected) for s in counts},conditioning='Final stage1 canonical-first-token plus full-answer success on template 0; not representative unconditional accuracy'))
  del e,tok;release()
 records=load(selection)['records'];evals=[r for r in records if r['split']=='evaluation']
 if not evals:dump(root/'TRAINING_COMPLETE.json',dict(status='no_final_evaluation_successes'));return
 # Final features first, then checkpoints in chronological order.
 ordered=[final]+[s for s in specs if s!=final]
 for s in ordered:
  folder=root/f'step_{s["step"]}'
  if (folder/'CAPTURE_COMPLETE.json').exists():continue
  tok,e=model_load(s,cfg)
  fingerprint=stamp(tok.backend_tokenizer.to_str());tokenfile=root/'tokenizer_fingerprint.json'
  if tokenfile.exists() and load(tokenfile)['fingerprint']!=fingerprint:raise ValueError('Tokenizer changed across training checkpoints')
  dump(tokenfile,dict(fingerprint=fingerprint))
  capture(e,tok,records,folder,cfg,templates=range(4));variance(evals,folder)
  dump(folder/'CAPTURE_COMPLETE.json',dict(status='complete'))
  del e,tok;release()
 summarize(root,cfg)
 dump(root/'TRAINING_COMPLETE.json',dict(status='complete'))


def summarize(root,cfg):
 root=Path(root);specs=load(root/'checkpoints.json');records=load(root/'selected_facts.json')['records'];ev=[r for r in records if r['split']=='evaluation'];finalroot=root/f'step_{specs[-1]["step"]}'
 rows=[];events=[];cross=[];movement=None
 for r in ev:
  final=load_arrays(cache_path(finalroot,r,0));correct=[];complete=True
  for s in specs:
   folder=root/f'step_{s["step"]}';m=read_meta(folder,r,0);correct.append(m.get('correct',False))
   if m.get('status')!='measured':complete=False;continue
   a=load_arrays(cache_path(folder,r,0))
   for l in range(len(a['gate'])):
    for si,site in enumerate(['subject','answer']):
     x=a['gate'][l,si].astype(float);y=final['gate'][l,si].astype(float);norm=np.linalg.norm(x)*np.linalg.norm(y)
     rows.append(dict(id=r['id'],group_id=r['group_id'],relation=r['relation'],step=s['step'],layer=l,site=site,correct=m['correct'],margin=m['margin'],subject_origin_margin=m['origins']['subject'],cosine_to_final=float(x@y/norm) if norm>1e-12 else None))
   if s==specs[0]:
    delta=np.abs(a['gate']-final['gate']).mean(1);movement=delta if movement is None else movement+delta
  ix=first_sustained(correct,cfg['training_sustained']) if complete else None
  events.append(dict(id=r['id'],group_id=r['group_id'],relation=r['relation'],complete_trajectory=complete,initial_correct=correct[0],final_correct=correct[-1],new_event=ix is not None and ix>0,first_sustained_step=specs[ix]['step'] if ix is not None else None,previous_observed_step=specs[ix-1]['step'] if ix is not None and ix>0 else None,initially_correct_left_censored=ix==0))
 for s in specs:
  file=root/f'step_{s["step"]}'/'reports/variance_interpretation.json'
  if file.exists():
   for x in load(file)['crossings']:cross.append(dict(step=s['step'],measurement=x['measurement'],between_layers=x['between_layers'],depth_fraction=(sum(x['between_layers'])/2+1)/len(final['gate'])))
 table(rows,root/'reports/training_trajectories.csv');table(events,root/'reports/training_events.csv');table(cross,root/'reports/training_crossovers.csv')
 # Optional gradient interface selects moving neurons descriptively, never invents gradients.
 if movement is not None:
  dump(root/'fixed_neurons.json',{'moving':dict(kind='largest_observed_gate_movement',sites=[dict(layer=l,neurons=np.argsort(-v)[:32].tolist()) for l,v in enumerate(movement)])})
 from gate_study.gradients import analyze
 manifest=cfg['gradient_manifest']
 if isinstance(manifest,dict):manifest=manifest.get(specs[-1]['repo'])
 analyze(manifest,root,dict(training_repo=specs[-1]['repo'],control_seeds=cfg['seeds']))
