"""CPU-only analysis of existing vectors; never substitutes norms for vectors."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
from channel_study.io import load,dump,digest
from .features import Crossed,Moments


def csv_write(path,rows):
 if not rows:return
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(previous,out,models=('llama1b','llama3b','llama8b','qwen7b')):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 if previous is None:
  dump(out/'status.json',dict(status='skipped',reason='No previous cache supplied; no GPU captures are launched.'));return
 previous=Path(previous)
 if not previous.is_dir():raise FileNotFoundError(previous)
 status=[];anova=[];sink=[];norms=[];variance=[]
 for model in models:
  root=previous/model;groups={};norm_groups={}
  for folder in sorted((root/'prompts').glob('*')):
   bp=folder/'baseline.json'
   if not bp.exists():continue
   p=load(bp)['prompt']
   if p.get('split') not in ('evaluation','eval') or p['template'] not in [0,1,2]:continue
   subject=str(p['group_id']);rel=p['relation'];fact=str(p['id']);t=p['template']
   groups.setdefault((rel,subject,fact),{})[t]=folder
   if (folder/'geometry.json').exists():
    for r in load(folder/'geometry.json'):
     norm_groups.setdefault((rel,t,r['layer']),[]).append((subject,float(r['first_norm'])))
  # Select a single fact per subject/relation BEFORE considering vector completeness.
  chosen={}
  for rel,subject,fact in sorted(groups):chosen.setdefault((rel,subject),fact)
  states={};complete=0;missing=0
  for (rel,subject),fact in sorted(chosen.items()):
   folders=groups[(rel,subject,fact)]
   if not all(t in folders and (folders[t]/'origin_vectors.pt').exists() for t in [0,1,2]):missing+=1;continue
   X=[torch.load(folders[t]/'origin_vectors.pt',map_location='cpu',weights_only=True).numpy() for t in [0,1,2]]
   if any(x.ndim!=3 or x.shape[1]!=2 for x in X):raise ValueError('Expected L x 2 origins x hidden origin_inputs cache')
   state=states.setdefault(rel,Crossed(out/'states'/model/(digest(rel)[:16]+'.npz')))
   # Crossed's generic gate slot is relabeled on output; these are input vectors, NOT gates.
   state.add(subject,{'gate':X});complete+=1
  for rel,state in states.items():
   for r in state.rows():
    r=dict(r);r['measurement']='origin_input_vector';r['origin']=['subject','template'][r['site']]
    anova.append(dict(model=model,relation=rel,**r))
  for (rel,t,l),values in norm_groups.items():
   # First average repeated facts within subject; then descriptive variation of scalar norms.
   by={}
   for subject,v in values:by.setdefault(subject,[]).append(v)
   x=np.array([np.mean(v) for v in by.values()]);mean=float(x.mean());sd=float(x.std())
   norms.append(dict(model=model,relation=rel,template=t,layer=l,subjects=len(x),mean_first_norm=mean,std_first_norm=sd,norm_cv=sd/mean if mean>1e-12 else None))
  # Newer reviewer captures contain exact vector sufficient statistics.
  for path in (root/'sink_state').glob('*.npz'):
   state=Moments(path)
   for r in state.rows():
    d=dict(r);den=d['mean_norm']
    d['vector_std_over_mean_vector_norm']=math.sqrt(d['trace_variance'])/den if den>1e-12 else None
    sink.append(dict(model=model,relation_key=path.stem,**d))
  for path in (root/'variance').glob('*.json'):
   d=load(path);variance.extend(dict(model=model,relation=d['relation'],**r) for r in d['rows'])
  status.append(dict(model=model,complete_origin_triplets=complete,missing_origin_triplets=missing,
   gate_attention_cache_available=bool(list((root/'variance').glob('*.json'))),sink_vector_cache_available=bool(list((root/'sink_state').glob('*.npz')))))
  print(model, status[-1],flush=True)
 csv_write(out/'origin_input_anova.csv',anova);csv_write(out/'first_norm_variation_PROXY.csv',norms)
 csv_write(out/'sink_vector_constancy.csv',sink);csv_write(out/'cached_gate_attention_anova.csv',variance)
 dump(out/'status.json',dict(models=status,interpretation={
  'anova':'Balanced within-relation subject x template ANOVA of cached origin INPUT vectors; NOT MLP gates or attention patterns. Interaction retained. Templates 0,1,2; one fact per subject/relation.',
  'norm_proxy':'STD(||first vector||)/mean(||first vector||) cannot establish vector constancy: a vector can rotate with constant norm.',
  'sink':'sqrt(E||v-Ev||^2)/||Ev|| uses newer sink_state captures only; undefined for zero mean. Qwen may have no distinct first-token origin when token zero belongs to the subject.',
  'missing':'Older split/geometry caches lack first-token vectors and actual gate/attention vectors. Full gate/sink tests need fresh GPU captures; not silently scheduled.'}))


def main():
 p=argparse.ArgumentParser();p.add_argument('--previous');p.add_argument('--out',required=True);a=p.parse_args();analyze(a.previous,a.out)
if __name__=='__main__':main()
