"""Post-hoc analyses use existing records only. Mean effects and mean sign-flip
statistics are aligned; no p-values silently attached to different estimands.
"""
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from .common import csv,dump,estimate,tests,group_label,regime
from causal_suite.stats import paired_p,auc_test


def verdict(r,tt,ss,share,q='full'):
 if r['q'][q]<=0:return 'source'
 return 'transport' if r['tau']<tt or share<ss else 'selection'


def reference(records):
 refs=[r for r in records if r.get('eligible') and r['split']=='calibration' and r.get('calibration_reference') and r['correct']]
 if not refs:raise ValueError('No saved calibration_reference successes. Do not substitute evaluation successes.')
 return refs


def posthoc(records,out,cfg):
 out.mkdir(parents=True,exist_ok=True);refs=reference(records);ev=[r for r in records if r.get('eligible') and r['split']=='evaluation'];fail=[r for r in ev if not r['correct']]
 origins=[];diagnostics=[];invariance=[];transitions=[];floor=cfg['threshold'];tt=float(np.quantile([r['tau'] for r in refs],floor))
 for r in ev:
  gold=r['ledger'][str(r['g'])]['origins'];wrong=r['ledger'][str(r['c'])]['origins'];m={k:gold[k]-wrong[k] for k in gold};err=abs(sum(m.values())-r['margin'])/max(1,abs(r['margin']))
  if err>cfg['certificate_tolerance']:raise AssertionError('Cached margin origins fail reconstruction')
  for origin,v in m.items():origins.append(dict(id=r['id'],group_id=r['group_id'],relation=r['relation'],label=group_label(r),regime=regime(r),origin=origin,margin_contribution=v,total_margin=r['margin'],correct=r['correct'],certificate=err))
  if r['correct']:
   for convention in ['full','hard20']:
    ss=float(np.quantile([a['share'] for a in refs],floor));lab=verdict(r,tt,ss,r['share'],convention)
    diagnostics.append(dict(id=r['id'],group_id=r['group_id'],relation=r['relation'],convention=convention,diagnostic_without_correct_override=lab,source_deficient=r['q'][convention]<=0,transport_deficient=r['tau']<tt or r['share']<ss,tau=r['tau'],q=r['q'][convention]))
 for shift in cfg['shifts']:
  # Output-bias shift is external: ALL logits receive the same constant.
  # Original source and other input-origin contributions are unchanged; add shift to bias.
  if any(abs(a['gold_logit']+shift)<1e-6 for a in refs):
   invariance.append(dict(shift=shift,status='calibration_singular'));continue
  shares=[a['tau']/abs(a['gold_logit']+shift) for a in refs];ss=float(np.quantile(shares,floor))
  for r in ev:
   near=abs(r['gold_logit']+shift)<1e-6
   lab='undefined' if near else verdict(r,tt,ss,r['tau']/abs(r['gold_logit']+shift))
   original=verdict(r,tt,float(np.quantile([a['share'] for a in refs],floor)),r['share'])
   transitions.append(dict(id=r['id'],group_id=r['group_id'],shift=shift,correct=r['correct'],original=original,shifted=lab,changed=lab!=original,singular=near))
  subset=[a for a in transitions if a['shift']==shift and not a['correct'] and not a['singular']]
  invariance.append(dict(shift=shift,status='valid',calibration_n=len(refs),share_threshold=ss,**estimate([a['changed'] for a in subset],[a['group_id'] for a in subset],cfg)))
 csv(origins,out/'contrastive_origins.csv');csv(diagnostics,out/'correct_but_deficient.csv');csv(invariance,out/'logit_shift_summary.csv');csv(transitions,out/'logit_shift_transitions.csv')
 summaries=[]
 d=pd.DataFrame(origins)
 for (label,origin),s in d.groupby(['label','origin']):summaries.append(dict(label=label,origin=origin,**estimate(s.margin_contribution,s.group_id,cfg)))
 csv(summaries,out/'contrastive_origin_intervals.csv')
 ds=pd.DataFrame(diagnostics);stats=[]
 if len(ds):
  for convention,s in ds.groupby('convention'):
   for metric in ['source_deficient','transport_deficient']:stats.append(dict(convention=convention,metric=metric,**estimate(s[metric],s.group_id,cfg)))
 csv(stats,out/'correct_deficiency_rates.csv')
 matched,mstats=matched_analysis(fail,cfg);csv(matched,out/'matched_pairs.csv');tests(mstats,out/'matched_effects_holm.csv')
 dump(out/'analysis_notes.json',dict(calibration_n=len(refs),evaluation_eligible=len(ev),failures=len(fail),
  shift='Exact invariance of softmax and argmax to common logit shift; recalibration uses same independent successful cases. Gold margin and contrastive origin contributions are unchanged algebraically.',
  matching='Maximum-cardinality minimum-cost selection/transport matching without replacement within relation, after keeping one deterministic fact per subject, with explicit margin and prompt-length calipers. Rank included only when present. No matches outside calipers.',
  interpretation='Source readability and transport thresholds are observational measurements. A correct answer may violate either. No absent/overridden/misretrieved causal relabelling is assumed.'))
 plots(out)


def matched_analysis(fail,cfg):
 # One fact per subject, chosen without reference to outcomes or matching quality.
 unique={}
 for r in sorted(fail,key=lambda r:str(r['id'])):unique.setdefault(r['group_id'],r)
 fail=list(unique.values())
 rows=[];summaries=[]
 for relation in sorted({r['relation'] for r in fail}):
  a=[r for r in fail if group_label(r)=='selection' and r['relation']==relation];b=[r for r in fail if group_label(r)=='transport' and r['relation']==relation]
  if not a or not b:continue
  cost=np.full((len(a),len(b)),1e9);rank_available=all('target_rank' in r for r in a+b)
  for i,x in enumerate(a):
   for j,y in enumerate(b):
    dm=abs(x['margin']-y['margin']);dl=abs(len(x['input_ids'])-len(y['input_ids']));dr=abs(x['target_rank']-y['target_rank']) if rank_available else 0
    if x['group_id']!=y['group_id'] and dm<=cfg['match_margin_caliper'] and dl<=cfg['match_length_caliper'] and dr<=cfg['match_rank_caliper']:
     cost[i,j]=dm/max(cfg['match_margin_caliper'],1e-9)+dl/max(cfg['match_length_caliper'],1)+dr/max(cfg['match_rank_caliper'],1)
  ii,jj=linear_sum_assignment(cost)
  for i,j in zip(ii,jj):
   if cost[i,j]>=1e9:continue
   x,y=a[i],b[j];rows.append(dict(selection_id=x['id'],transport_id=y['id'],relation=relation,selection_margin=x['margin'],transport_margin=y['margin'],
    selection_length=len(x['input_ids']),transport_length=len(y['input_ids']),rank_matched=rank_available,
    competitor_share_difference=x['competitor_share']-y['competitor_share'],
    competitor_absolute_share_difference=x['competitor_absolute_share']-y['competitor_absolute_share'],
    subject_margin_difference=(x['ledger'][str(x['g'])]['origins']['subject']-x['ledger'][str(x['c'])]['origins']['subject'])-(y['ledger'][str(y['g'])]['origins']['subject']-y['ledger'][str(y['c'])]['origins']['subject'])))
 for metric in ['competitor_share_difference','competitor_absolute_share_difference','subject_margin_difference']:
  values=[r[metric] for r in rows];groups=[str(i) for i in range(len(rows))]
  summaries.append(dict(metric=metric,**estimate(values,groups,cfg),p=paired_p(values,groups,cfg['permutations'],cfg['seed'])))
 return rows,summaries


def plots(out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 d=pd.read_csv(out/'contrastive_origins.csv');a=d.groupby(['label','origin']).margin_contribution.mean().unstack().fillna(0)
 fig,ax=plt.subplots(figsize=(7,4));pos=np.zeros(len(a));neg=pos.copy()
 for name in a:
  y=a[name].values;ax.bar(a.index,y,bottom=np.where(y>=0,pos,neg),label=name);pos+=np.maximum(y,0);neg+=np.minimum(y,0)
 ax.axhline(0,color='black',lw=.6);ax.set_ylabel('Mean gold-minus-competitor contribution');ax.legend();fig.tight_layout();fig.savefig(out/'contrastive_origins.pdf');plt.close(fig)
 s=pd.read_csv(out/'logit_shift_summary.csv')
 if 'estimate' in s:
  fig,ax=plt.subplots(figsize=(6,4));ax.plot(s['shift'],s.estimate,'o-');ax.fill_between(s['shift'],s.lo,s.hi,alpha=.2);ax.set(xlabel='Constant added to every logit',ylabel='Fraction of failure diagnoses changed');fig.tight_layout();fig.savefig(out/'logit_shift_sensitivity.pdf');plt.close(fig)


def matched_repairs(records,repair_csv,out,cfg,model,mode):
 """Same repair arm/layer/dose on matched subjects; exploratory differences in
 response, never compare unlike units. Missing intervention arms remain missing.
 """
 from pathlib import Path
 path=Path(repair_csv)
 if not path.exists():
  dump(out/'matched_repair_status.json',{'status':'not_run','reason':'Input archive has no reports/repairs.csv'});return
 pairs,_=matched_analysis([r for r in records if r.get('eligible') and r['split']=='evaluation' and not r['correct']],cfg)
 d=pd.read_csv(path,dtype={'id':str});d=d[(d.model==model)&(d['mode']==mode)&(d.plant=='natural')].copy();d['layer']=d.layer.fillna(-1)
 values=[]
 keys=['family','arm','dose','layer']
 for i,pair in enumerate(pairs):
  a=d[d.id==str(pair['selection_id'])];b=d[d.id==str(pair['transport_id'])]
  merged=a.merge(b,on=keys,suffixes=('_selection','_transport'),validate='one_to_one')
  for _,row in merged.iterrows():values.append(dict(pair=i,**{k:row[k] for k in keys},margin_gain_difference=row.margin_gain_selection-row.margin_gain_transport,recovery_difference=float(row.recovered_selection)-float(row.recovered_transport)))
 csv(values,out/'matched_repair_pairs.csv');summary=[]
 if values:
  for key,s in pd.DataFrame(values).groupby(keys):
   for metric in ['margin_gain_difference','recovery_difference']:
    summary.append(dict(zip(keys,key),metric=metric,**estimate(s[metric],s.pair,cfg),p=paired_p(s[metric],s.pair,cfg['permutations'],cfg['seed'])))
 tests(summary,out/'matched_repair_effects_holm.csv')
