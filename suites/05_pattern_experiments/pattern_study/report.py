from pathlib import Path
import numpy as np
import pandas as pd
from causal_suite.stats import boot,holm
from .common import load,dump,table


def sign_test(v,groups,n,seed):
 """Cluster sign-flip, exact when <=16 independent clusters; otherwise Monte Carlo."""
 sums=pd.Series(np.asarray(v,float)).groupby(np.asarray(groups)).sum().to_numpy();observed=abs(sums.sum())
 if not len(sums):return np.nan
 if len(sums)<=16:
  codes=np.arange(2**len(sums),dtype=np.uint64)[:,None];bits=(codes>>np.arange(len(sums),dtype=np.uint64))&1
  return float(np.mean(np.abs((bits.astype(float)*2-1)@sums)>=observed-1e-12))
 rng=np.random.default_rng(seed);count=0;done=0
 while done<n:
  k=min(2048,n-done);draw=rng.integers(0,2,(k,len(sums)),dtype=np.int8)*2-1
  count+=int((np.abs(draw@sums)>=observed-1e-12).sum());done+=k
 return (count+1)/(n+1)


def summarize_weights(root,cfg):
 rows=[]
 for p in (Path(root)/'weights').glob('*/*.json'):
  d=load(p)
  if d.get('status')!='measured':continue
  checks=d['specificity'];keep=[c for c in checks if c['baseline_top1']]
  rows.append({**{k:d[k] for k in ['pair_id','id','group_id','relation','recipient_template','donor_template','scope','rank_requested','arm','seed','rho','actual_delta_norm','live_pattern_error']},**d['endpoint'],
   'margin_gain':d['margin_gain'],'specificity_n':len(checks),'probe_kl':np.mean([c['kl'] for c in checks]) if checks else np.nan,'probe_prediction_change':np.mean([c['prediction_changed'] for c in checks]) if checks else np.nan,'probe_top1_retention':np.mean([c['gold_top1'] for c in keep]) if keep else np.nan})
 out=Path(root)/'reports';w=table(rows,out/'weights.csv')
 if not len(w):return
 metrics=['margin_gain','gold_top1','full_answer_correct','probe_kl','probe_prediction_change','probe_top1_retention']
 # Equal fact weighting: average repeated donor/recipient template pairs and seeds first.
 avg=w.groupby(['id','group_id','scope','rank_requested','arm','rho'],as_index=False)[metrics].mean();summary=[]
 for k,s in avg.groupby(['scope','rank_requested','arm','rho']):
  for metric in metrics:
   est=boot(s[metric],s.group_id,n=cfg['bootstrap'],seed=cfg['seed'])
   if est['lo']==est['hi']:est['lo']=est['hi']=None
   summary.append(dict(zip(['scope','rank','arm','rho'],k),metric=metric,**est))
 table(summary,out/'weight_intervals.csv');comparisons=[]
 for k,s in avg.groupby(['scope','rank_requested','rho']):
  for metric in ['margin_gain','gold_top1','probe_kl']:
   wide=s.pivot(index=['id','group_id'],columns='arm',values=metric)
   for target,control in [('pooled_with_target','gradient_gate'),('pooled_with_target','gradient_down'),('pooled_with_target','random_gate'),('gradient_gate','gradient_down'),('gradient_gate','gradient_up')]:
    if target not in wide or control not in wide:continue
    both=wide[[target,control]].dropna();v=both[target]-both[control];groups=both.index.get_level_values('group_id')
    est=boot(v,groups,n=cfg['bootstrap'],seed=cfg['seed'])
    comparisons.append(dict(zip(['scope','rank','rho'],k),metric=metric,target=target,control=control,**est,p=sign_test(v,groups,cfg['permutations'],cfg['seed'])))
 if comparisons:
  tests=pd.DataFrame(comparisons);tests['p_holm']=holm(tests.p);table(tests,out/'weight_paired_tests.csv')
  dump(out/'inference_resolution.json',dict(comparisons=len(tests),permutations=cfg['permutations'],monte_carlo_min_p=1/(cfg['permutations']+1),first_holm_threshold=.05/len(tests),sufficient_resolution=1/(cfg['permutations']+1)<=.05/len(tests),note='Correct family retained; do not reduce it after inspecting outcomes. Degenerate bootstrap intervals omitted.'))


def summarize_curves(root):
 rows=[];half=[];variants=[];sink=[];out=Path(root)/'reports'
 for p in (Path(root)/'curves').glob('*/*.json'):
  d=load(p)
  if 'pair_id' not in d:continue
  base={k:d.get(k) for k in ['pair_id','id','group_id','relation','direction','operation','scope','order','seed','mode']}
  if 'rows' in d:
   rr=d['rows'];rows.extend(dict(**base,**r) for r in rr);total=rr[-1]['margin_change']
   hits=[r['k'] for r in rr if total*r['margin_change']>=.5*total**2] if abs(total)>1e-6 else []
   half.append(dict(**base,full_effect=total,k_half=min(hits) if hits else None,total_coordinates=rr[-1]['k'],interpretation='First observed crossing on sampled k grid; signed half endpoint, no monotonicity assumption'))
  elif 'endpoint' in d:variants.append(dict(**base,**d['endpoint'],baseline_margin=d['baseline']['margin'],margin_change=d['margin_change'],donor_gate=d.get('donor_gate'),donor_attention=d.get('donor_attention'),**{f'origin_{k}':v for k,v in d.get('origin_margin',{}).items()}))
 for p in (Path(root)/'sink').glob('*.json'):
  d=load(p)
  if d.get('status')=='measured':sink.append(dict(**{k:d[k] for k in ['id','group_id','relation','head_kind','seed','dose','margin_gain']},**d['endpoint']))
 table(rows,out/'curves.csv');table(half,out/'half_effect_sizes.csv');table(variants,out/'sign_magnitude_variants.csv');table(sink,out/'sink_sweep.csv')


def plots(root):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 p=Path(root)/'reports'
 def read(n):
  try:return pd.read_csv(p/n)
  except (FileNotFoundError,pd.errors.EmptyDataError):return pd.DataFrame()
 v=read('variance_components.csv')
 if len(v):
  fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
  for ax,name in zip(axes,['subject_gate','answer_gate','attention_roles']):
   s=v[v.measurement==name].groupby('layer')[['subject_ss','template_ss','interaction_ss','total_ss']].sum()
   for term in ['subject','template','interaction']:ax.plot(s.index,s[term+'_ss']/s.total_ss,label=term)
   ax.set(title=name,xlabel='Layer',ylim=(0,1));ax.legend()
  fig.tight_layout();fig.savefig(p/'variance.pdf');plt.close(fig)
 w=read('weight_intervals.csv')
 if len(w):
  for scope,s in w[(w.metric=='gold_top1')].groupby('scope'):
   fig,ax=plt.subplots(figsize=(8,4))
   for (arm,rank),a in s.groupby(['arm','rank']):
    if arm not in ['pooled_with_target','gradient_gate','gradient_down']:continue
    ax.plot(a.rho,a.estimate,'o-',label=f'{arm} rank {rank}')
   ax.set(xscale='log',xlabel='Relative weight budget',ylabel='Gold top-1 fraction',title=scope);ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(p/f'weight_{scope}.pdf');plt.close(fig)
 curves=read('curves.csv')
 if len(curves):
  for (direction,op,mode),s in curves[curves.scope=='all'].groupby(['direction','operation','mode']):
   fig,ax=plt.subplots(figsize=(6,4))
   for order,g in s.groupby('order'):
    a=g.groupby(['id','k']).margin_change.mean().groupby('k').mean();ax.plot(a.index,a.values,'o-',label=order)
   ax.set(xscale='symlog',xlabel='Reassigned coordinates',ylabel='Mean margin change',title=f'{direction}: {op}, {mode}');ax.legend();fig.tight_layout();fig.savefig(p/f'curve_{direction}_{op}_{mode}.pdf');plt.close(fig)


def run(root,cfg):
 summarize_weights(root,cfg);summarize_curves(root);plots(root)
