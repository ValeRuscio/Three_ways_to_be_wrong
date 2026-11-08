from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .common import table,dump,estimate
from causal_suite.stats import paired_p,holm


def intervals(rows,group_cols,metrics,cfg):
 result=[]
 if not len(rows):return pd.DataFrame()
 for key,s in rows.groupby(group_cols,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  for metric in metrics:
   est=estimate(s[metric],s.group_id,cfg)
   # Nonparametric bootstrap cannot express uncertainty beyond an all-zero or
   # all-one observed sample. Do not print degenerate intervals as certainty.
   if est['lo']==est['hi']:est['lo']=None;est['hi']=None
   result.append(dict(zip(group_cols,key),metric=metric,**est))
 return pd.DataFrame(result)


def paired_controls(d,group_cols,dose_col,target,controls,metrics,cfg):
 results=[]
 if not len(d):return pd.DataFrame()
 keys=['id','group_id']+group_cols+[dose_col,'arm']
 avg=d.groupby(keys,dropna=False,as_index=False)[metrics].mean() # average seeds per recipient
 for key,s in avg.groupby(group_cols+[dose_col],dropna=False):
  if not isinstance(key,tuple):key=(key,)
  for metric in metrics:
   w=s.pivot(index=['id','group_id'],columns='arm',values=metric)
   if target not in w:continue
   for control in controls:
    if control not in w:continue
    both=w[[target,control]].dropna();v=both[target]-both[control];groups=both.index.get_level_values('group_id')
    results.append(dict(zip(group_cols+[dose_col],key),target=target,control=control,metric=metric,**estimate(v,groups,cfg),p=paired_p(v,groups,cfg['permutations'],cfg['seed'])))
 result=pd.DataFrame(results)
 if len(result):result['p_holm']=holm(result.p)
 return result


def distance_models(d,out,cfg):
 result=[];primary=d[d.arm=='donor_preactivation']
 for key,s in primary.groupby(['set_kind','site','alpha']):
  s=s.drop_duplicates('id');n=len(s)
  # Continuous distance, baseline difficulty, perturbation magnitude and relation.
  if n<5:result.append(dict(set_kind=key[0],site=key[1],alpha=key[2],n=n,status='insufficient_cases'));continue
  x=s.signed_distance_mean.to_numpy(float);y=s.top1_margin_gain.to_numpy(float);rho=spearmanr(x,y).statistic
  nuisance=np.column_stack([s.baseline_margin,np.log1p(s.baseline_rank),s.preactivation_delta_norm,pd.get_dummies(s.relation,drop_first=True,dtype=float)])
  X=np.column_stack([np.ones(n),x,nuisance]);rank=np.linalg.matrix_rank(X)
  row=dict(set_kind=key[0],site=key[1],alpha=key[2],n=n,spearman_distance_vs_gain=rho,status='descriptive_only')
  if n> X.shape[1]+5 and rank==X.shape[1]:
   coef=np.linalg.lstsq(X,y,rcond=None)[0][1];rng=np.random.default_rng(cfg['seed']);samples=[];groups=s.group_id.to_numpy();unique=np.unique(groups)
   for _ in range(cfg['bootstrap']):
    draw=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
    if np.linalg.matrix_rank(X[draw])==X.shape[1]:samples.append(np.linalg.lstsq(X[draw],y[draw],rcond=None)[0][1])
   row.update(status='adjusted_OLS_association',distance_coefficient=coef,lo=np.quantile(samples,.025) if samples else None,hi=np.quantile(samples,.975) if samples else None,valid_bootstrap=len(samples))
  result.append(row)
 table(result,out/'distance_response_models.csv')


def summarize(root,cfg):
 root=Path(root);out=root/'reports';out.mkdir(parents=True,exist_ok=True);weight=[];specific=[];coverage=[];activation=[];route=[];certs=[]
 for p in (root/'features').glob('*/*.json'):
  m=json.loads(p.read_text());coverage.append({k:m.get(k) for k in ['id','group_id','relation','split','template','status','correct','label']})
  if 'certificate' in m:certs.append(dict(id=m['id'],template=m['template'],native=m['certificate'],factor=m['factor_certificate']))
 table(coverage,out/'prompt_coverage.csv');table(certs,out/'certificates.csv')
 wcoverage=[]
 for p in (root/'weight_edits').glob('*/*.json'):
  d=json.loads(p.read_text())
  if p.name=='coverage.json':wcoverage.append(d);continue
  small={k:v for k,v in d.items() if not isinstance(v,(dict,list))};small['baseline_margin']=d['baseline']['top1_margin'];small['specificity_n']=len(d['specificity'])
  checks=d['specificity']
  small['mean_specificity_kl']=np.mean([x['kl'] for x in checks]) if checks else np.nan
  small['specificity_prediction_change']=np.mean([x['prediction_changed'] for x in checks]) if checks else np.nan
  keep=[x for x in checks if x['baseline_top1']]
  small['specificity_top1_retention']=np.mean([x['gold_top1'] for x in keep]) if keep else np.nan
  weight.append(small)
  for x in checks:specific.append(dict(target_id=d['id'],set_kind=d['set_kind'],site=d['site'],arm=d['arm'],seed=d['seed'],rho=d['rho'],**x))
 w=table(weight,out/'weight_edits.csv');table(specific,out/'specificity.csv');dump(out/'weight_coverage.json',wcoverage)
 if len(w):
  metrics=['top1_margin_gain','fixed_margin_gain','gold_top1','final_normalized_relative_change','mean_specificity_kl','specificity_prediction_change','specificity_top1_retention']
  avg=w.groupby(['id','group_id','label','set_kind','site','rho','arm'],as_index=False)[metrics].mean()
  table(intervals(avg,['label','set_kind','site','rho','arm'],metrics,cfg),out/'weight_intervals.csv')
  table(paired_controls(w,['label','set_kind','site'],'rho','gate_target',['down_target','up_target','gate_reverse','gate_random_neurons','gate_random_direction'],['top1_margin_gain','gold_top1','mean_specificity_kl'],cfg),out/'weight_paired_tests_holm.csv')
 for p in (root/'activation_patches').glob('*.json'):activation+=json.loads(p.read_text())['rows']
 a=table(activation,out/'activation_patches.csv')
 if len(a):
  table(paired_controls(a,['label','set_kind','site'],'alpha','donor_preactivation',['random_neurons','random_direction'],['top1_margin_gain','gold_top1'],cfg),out/'activation_paired_tests_holm.csv');distance_models(a,out,cfg)
 for p in (root/'routing').glob('*.json'):route+=json.loads(p.read_text())['rows']
 rt=table(route,out/'routing.csv')
 if len(rt):
  good=rt[rt.status=='measured'].copy()
  if len(good):
   good['donor_condition']=np.where(good.donor_template!=0,'same_subject_different_template',np.where(good.donor_correct,'same_template_other_success','same_template_other_failure'))
   avg=good.groupby(['id','group_id','label','operation','head_kind','donor_condition'],as_index=False)[['top1_margin_gain','gold_top1','baseline_sink_share']].mean()
   table(intervals(avg,['label','operation','head_kind','donor_condition'],['top1_margin_gain','gold_top1','baseline_sink_share'],cfg),out/'routing_intervals.csv')
   associations=[]
   for k,s in avg.groupby(['operation','head_kind','donor_condition']):
    associations.append(dict(operation=k[0],head_kind=k[1],donor_condition=k[2],n=len(s),sink_vs_gain_spearman=spearmanr(s.baseline_sink_share,s.top1_margin_gain).statistic if len(s)>2 else None))
   table(associations,out/'sink_routing_associations.csv')
 dump(out/'interpretation.json',dict(
  factorization='g * (W_up x_subject_origin) * (W_down^T r) is exact for a local frozen MLP edge. It is not a disjoint decomposition summed across layers, nor a unique factorization of live nonlinear causation.',
  selection='Independent relation-support sets and same-fact donor-support sets remain separate. Positive top-k selection is an operational definition with k sensitivity, not proof of fact storage.',
  distance='SiLU is smooth, can be negative, and has no closed/open boundary. Distances to -2,0,2 preactivation planes are explicitly chosen geometric probes; no binary failure cause follows.',
  edits='Live temporary weight edits with rollback. Equal Frobenius norms do not equate functional costs; report output displacement and specificity alongside efficacy. Rank-one down editing is not an implementation of ROME.',
  endpoints='Gold top1 uses strongest current competitor. Fixed margin is secondary. Full-answer generation is conditional on canonical gold top1; null is unmeasured.',
  intervals='Subject-cluster bootstrap; random seeds and donors averaged per case. Degenerate bootstrap intervals omitted. Holm correction is within each reported comparison table.',
  routing='Pattern rows are interventions on probabilities, not isolated Q/K geometry. Role-mass transplants retain recipient within-role distributions. Sink association alone does not prove null-gate causation.'))
 plots(root,cfg)


def read_csv(path):
 try:return pd.read_csv(path)
 except (FileNotFoundError,pd.errors.EmptyDataError):return pd.DataFrame()


def plots(root,cfg=None):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(root)/'reports';v=read_csv(out/'variance_components.csv')
 if len(v):
  fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
  for ax,name in zip(axes,['subject_gate','answer_gate','attention_roles']):
   s=v[v.measurement==name].groupby('layer')[['subject_ss','template_ss','interaction_ss','total_ss']].sum()
   for part in ['subject','template','interaction']:ax.plot(s.index,s[part+'_ss']/s.total_ss,label=part)
   ax.set(title=name.replace('_',' '),xlabel='Layer',ylim=(0,1));ax.legend()
  axes[0].set_ylabel('Fraction of within-relation sum of squares');fig.tight_layout();fig.savefig(out/'gate_variance.pdf');plt.close(fig)
 prof=read_csv(out/'deficit_profiles.csv')
 if len(prof) and 'gate_reference_z' in prof:
  primary=prof[(prof.set_kind=='independent_relation') & (prof.k_requested==(cfg or {}).get('top_k',32))]
  for site,ss in primary.groupby('site'):
   fig,axes=plt.subplots(1,3,figsize=(12,3.5))
   for ax,metric in zip(axes,['gate_reference_z','up_subject_reference_z','down_gold_reference_z']):
    for label,s in ss.groupby('label'):
     avg=s.groupby('layer')[metric].mean();ax.plot(avg.index,avg.values,label=label)
    ax.axhline(0,color='k',lw=.5);ax.set(xlabel='Layer',ylabel=metric)
   axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/f'three_factor_profiles_{site}.pdf');plt.close(fig)
 w=read_csv(out/'weight_edits.csv')
 if len(w):
  for (kind,site),s in w.groupby(['set_kind','site']):
   fig,axes=plt.subplots(1,2,figsize=(10,4))
   for arm,a in s.groupby('arm'):
    if arm not in ['gate_target','down_target','up_target','gate_random_neurons','gate_random_direction']:continue
    avg=a.groupby(['id','rho'])[['gold_top1','mean_specificity_kl']].mean().groupby('rho').mean()
    axes[0].plot(avg.index,avg.gold_top1,'o-',label=arm);axes[1].plot(avg.mean_specificity_kl,avg.gold_top1,'o-',label=arm)
   axes[0].set(xscale='log',xlabel='Relative weight norm budget',ylabel='Gold top-1 fraction',ylim=(0,1));axes[1].set(xlabel='Mean KL on other subjects',ylabel='Gold top-1 fraction',ylim=(0,1));axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/f'weight_{kind}_{site}.pdf');plt.close(fig)
 a=read_csv(out/'activation_patches.csv')
 if len(a):
  a=a[a.arm=='donor_preactivation']
  for (kind,site),s in a.groupby(['set_kind','site']):
   fig,ax=plt.subplots(figsize=(6,4))
   for alpha,part in s.groupby('alpha'):ax.scatter(part.signed_distance_mean,part.top1_margin_gain,label=f'alpha={alpha}',alpha=.6)
   ax.set(xlabel='Mean signed distance to preactivation=0',ylabel='Live strongest-competitor margin gain');ax.legend();fig.tight_layout();fig.savefig(out/f'distance_{kind}_{site}.pdf');plt.close(fig)
 rt=read_csv(out/'routing_intervals.csv')
 if len(rt):
  s=rt[(rt.metric=='top1_margin_gain')&(rt.head_kind=='handoff')]
  if len(s):
   fig,ax=plt.subplots(figsize=(10,4));s.pivot_table(index=['label','operation'],columns='donor_condition',values='estimate').plot.bar(ax=ax);ax.axhline(0,color='k',lw=.5);ax.set_ylabel('Mean live top-1 margin gain');fig.tight_layout();fig.savefig(out/'routing_pattern_effects.pdf');plt.close(fig)


def training_plots(root):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(root)/'reports';d=read_csv(out/'training_event_aligned.csv')
 if not len(d):return
 for (kind,site),s in d.groupby(['set_kind','site']):
  fig,axes=plt.subplots(1,3,figsize=(12,3.5))
  for ax,metric in zip(axes,['pre','up_cosine','down_cosine']):
   a=s.groupby('relative_checkpoint')[metric].agg(['mean','count']);ax.plot(a.index,a['mean'],'o-');ax.axvline(0,color='k',ls=':');ax.set(xlabel='Checkpoint index relative to first sustained success',ylabel=metric)
  fig.tight_layout();fig.savefig(out/f'training_{kind}_{site}.pdf');plt.close(fig)
