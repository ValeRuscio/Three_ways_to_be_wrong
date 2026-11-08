from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import csv,dump,estimate,tests
from causal_suite.stats import paired_p


def report(root,cfg):
 root=Path(root);out=root/'reports';out.mkdir(exist_ok=True)
 hybrids=[];edits=[];curves=[];coverage=[];geometry=[]
 for p in sorted((root/'hybrids').glob('*.json')):
  obj=json.loads(p.read_text());coverage.append(dict(id=obj['id'],stage='hybrids',success_donors=obj['success_donors'],failure_donors=obj['failure_donors']));hybrids+=obj['rows']
 if hybrids:
  d=csv(hybrids,out/'hybrids.csv');factor=[]
  keys=['id','group_id','label','regime','donor_id','donor_type','normalization']
  for key,s in d.groupby(keys):
   v={(int(r.attention),int(r.gates)):r.margin for r in s.itertuples()}
   if len(v)!=4:continue
   factor.append(dict(zip(keys,key),A_effect=v[1,0]-v[0,0],G_effect=v[0,1]-v[0,0],interaction=v[1,1]-v[1,0]-v[0,1]+v[0,0],joint_gain=v[1,1]-v[0,0]))
  f=csv(factor,out/'hybrid_factorial_per_pair.csv');summary=[]
  # Average donor repetitions first. Recipient subject is the bootstrap unit.
  if len(f):
   metricnames=['A_effect','G_effect','interaction','joint_gain']
   avg=f.groupby(['id','group_id','label','regime','donor_type','normalization'],as_index=False)[metricnames].mean()
   for (label,dt,norm),s in avg.groupby(['label','donor_type','normalization']):
    for metric in metricnames:
     summary.append(dict(label=label,donor_type=dt,normalization=norm,metric=metric,**estimate(s[metric],s.group_id,cfg),p=paired_p(s[metric],s.group_id,cfg['permutations'],cfg['seed'])))
   tests(summary,out/'hybrid_factorial_summary_holm.csv')
   # Success donors versus failure donors, matched on recipient.
   paired=[]
   for (label,norm),s in avg.groupby(['label','normalization']):
    for metric in metricnames:
     wide=s.pivot(index=['id','group_id'],columns='donor_type',values=metric)
     if {'success','failure_control'}<=set(wide.columns):
      wide=wide.dropna();delta=wide.success-wide.failure_control
      paired.append(dict(label=label,normalization=norm,metric=metric,**estimate(delta,wide.index.get_level_values('group_id'),cfg),p=paired_p(delta,wide.index.get_level_values('group_id'),cfg['permutations'],cfg['seed'])))
   tests(paired,out/'hybrid_success_vs_failure_donor_holm.csv')
 for p in sorted((root/'edit_search').glob('*/*.json')):
  if p.name=='donor.json':
   obj=json.loads(p.read_text());coverage.append(dict(id=obj['id'],stage='paraphrase',verified=obj['verified'],initial_margin=obj.get('initial_margin')));continue
  obj=json.loads(p.read_text());meta={k:obj[k] for k in ['id','group_id','label','regime','mode','control']}
  edits.append(dict(**meta,stop=obj['stop'],recovered_margin=obj['recovered_margin'],gold_top1=obj['gold_top1'],edit_count=obj['edit_count'],pruned_count=obj['pruned_count'],initial_margin=obj['initial_margin']))
  for step in range(cfg['edit_budget']+1):
   point=obj['trace'][min(step,len(obj['trace'])-1)]
   curves.append(dict(**meta,budget=step,margin=point['margin'],margin_gain=point['margin']-obj['initial_margin'],recovered=point['margin_positive'],gold_top1=point['gold_top1'],observed_edits=point['edits']))
 if edits:
  e=csv(edits,out/'edit_search_summary.csv');c=csv(curves,out/'edit_search_curves.csv');summary=[]
  for key,s in e.groupby(['label','mode','control']):
   for metric in ['recovered_margin','gold_top1']:
    summary.append(dict(zip(['label','mode','control'],key),metric=metric,**estimate(s[metric],s.group_id,cfg)))
  csv(summary,out/'edit_recovery_intervals.csv')
  cc=[]
  for key,s in c.groupby(['label','mode','control','budget']):
   cc.append(dict(zip(['label','mode','control','budget'],key),n=len(s),mean_margin_gain=s.margin_gain.mean(),median_margin_gain=s.margin_gain.median(),q25=s.margin_gain.quantile(.25),q75=s.margin_gain.quantile(.75),recovery_fraction=s.recovered.mean(),top1_fraction=s.gold_top1.mean()))
  csv(cc,out/'edit_curves_aggregate.csv')
 for p in sorted((root/'geometry').glob('*.json')):geometry+=json.loads(p.read_text())['rows']
 if geometry:
  geo=csv(geometry,out/'gradient_geometry.csv');summary=[]
  for key,s in geo.groupby(['label','layer','head','basis','rank']):
   summary.append(dict(zip(['label','layer','head','basis','rank'],key),**estimate(s.local_energy_fraction,s.group_id,cfg)))
  csv(summary,out/'gradient_geometry_intervals.csv')
 csv(coverage,out/'coverage.csv')
 dump(out/'interpretation.json',dict(
  hybrids='Frozen counterfactual computations; attention, gates and normalization are coupled in a live network. Gates do not isolate stored content. Positive fixed-competitor margin is not necessarily a correct answer.',
  edit_search='Greedy verified upper bounds, with deletion pruning, not minimum edit counts or information-theoretic bits. Single-edge operations redistribute other row entries. No-improving-single-edit does not rule out interacting combinations. Curves plateau after termination; denominators include all verified-donor searches. Generalization beyond donor-available failures is unsupported.',
  sink='First-input-position origin and attention sink mass are separate measurements. Low within-template variance plus frequency alignment is descriptive evidence, not proof of a prior channel. Frequency association omitted without independently supplied counts.',
  geometry='Uses per-head output-coordinate gradient Gram subspaces, not flattened parameter-space eigenspaces. Local head deliveries overlap across computational paths; do not sum them as disjoint origins. No pretraining claims from fresh inference-time gradients.'))
 plots(out)


def plots(out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 p=out/'hybrid_factorial_summary_holm.csv'
 if p.exists() and p.stat().st_size>2:
  d=pd.read_csv(p);d=d[(d.donor_type=='success')&(d.normalization=='recipient')]
  if len(d):
   fig,ax=plt.subplots(figsize=(8,4));d.pivot(index='label',columns='metric',values='estimate').plot.bar(ax=ax);ax.axhline(0,color='k',lw=.6);ax.set_ylabel('Mean frozen margin change');fig.tight_layout();fig.savefig(out/'hybrid_factorial.pdf');plt.close(fig)
 p=out/'edit_curves_aggregate.csv'
 if p.exists():
  d=pd.read_csv(p)
  for mode,s in d[d.control=='greedy'].groupby('mode'):
   fig,axes=plt.subplots(1,2,figsize=(11,4))
   for label,a in s.groupby('label'):
    axes[0].plot(a.budget,a.median_margin_gain,label=label);axes[0].fill_between(a.budget,a.q25,a.q75,alpha=.15)
    axes[1].plot(a.budget,a.recovery_fraction,label=label)
   axes[0].set(xlabel='Allowed edit operations',ylabel='Margin gain: median and IQR');axes[1].set(xlabel='Allowed edit operations',ylabel='Fraction with positive fixed margin',ylim=(0,1));axes[0].legend();fig.tight_layout();fig.savefig(out/f'edit_distance_{mode}.pdf');plt.close(fig)
 p=out/'sink_template_variation.csv'
 if p.exists() and p.stat().st_size>2:
  d=pd.read_csv(p);fig,ax=plt.subplots(figsize=(7,4));d.pivot_table(index='relation',columns='origin',values='variation_fraction').plot.bar(ax=ax);ax.set_ylabel('Within-template variance / second moment');fig.tight_layout();fig.savefig(out/'sink_variation.pdf');plt.close(fig)
  if d.frequency_cosine_of_mean.notna().any():
   fig,ax=plt.subplots(figsize=(6,4))
   for name,s in d.groupby('origin'):ax.scatter(s.variation_fraction,s.frequency_cosine_of_mean,label=name)
   ax.set(xlabel='Within-template variance / second moment',ylabel='Mean-vector cosine with log frequency');ax.legend();fig.tight_layout();fig.savefig(out/'sink_frequency.pdf');plt.close(fig)
