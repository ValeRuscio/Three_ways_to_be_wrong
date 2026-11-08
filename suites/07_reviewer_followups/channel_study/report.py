"""Descriptive summaries; subject-cluster bootstrap after equal-fact averaging.
No uncorrected search for best layer/dose, and no causal-stage interpretation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .io import load,dump


def summarize(df,keys,metrics,cfg):
 if df.empty:return pd.DataFrame()
 metrics=[m for m in metrics if m in df]
 # First repetitions within prompt, then templates within fact. Each fact equal weight.
 prompt=df.groupby(keys+['id','group_id','template'],dropna=False)[metrics].mean().reset_index()
 fact=prompt.groupby(keys+['id','group_id'],dropna=False)[metrics].mean().reset_index()
 rows=[];rng=np.random.default_rng(cfg['seed'])
 for key,g in fact.groupby(keys,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  row=dict(zip(keys,key));subjects=g.group_id.unique();row.update(facts=len(g),subjects=len(subjects))
  for m in metrics:
   valid=g.dropna(subset=[m]);row[m]=float(valid[m].mean()) if len(valid) else None
   row[m+'_valid_facts']=len(valid)
   clusters=[v[m].to_numpy() for _,v in valid.groupby('group_id')]
   if len(clusters)>=5:
    sums=np.array([v.sum() for v in clusters]);counts=np.array([len(v) for v in clusters])
    ix=rng.integers(len(clusters),size=(cfg['bootstrap'],len(clusters)))
    vals=sums[ix].sum(1)/counts[ix].sum(1)
    row[m+'_lo'],row[m+'_hi']=np.quantile(vals,[.025,.975]).tolist()
   else:row[m+'_lo']=row[m+'_hi']=None
  rows.append(row)
 return pd.DataFrame(rows)


def report(root,cfg):
 root=Path(root);out=root/'reports';out.mkdir(parents=True,exist_ok=True)
 allrows={s:[] for s in ['split','fisher','geometry','perturb','frontier','training']};coverage=[]
 for model in root.iterdir():
  if not model.is_dir() or model.name.startswith('training_'):continue
  for folder in (model/'prompts').glob('*'):
   if not (folder/'baseline.json').exists():continue
   b=load(folder/'baseline.json');p=b['prompt'];meta={k:p[k] for k in ['id','group_id','template','relation']}
   meta.update(model=model.name,regime=b['regime'],baseline_answer_correct=b['answer_correct'])
   coverage.append(dict(meta,canonical_top1=b['canonical_top1']))
   for stage in ['split','fisher','geometry','perturb','frontier']:
    if not (folder/(stage+'_DONE.json')).exists():continue
    if stage in ['split','fisher','geometry']:records=load(folder/(stage+'.json'))
    else:records=[load(f) for f in sorted((folder/stage).glob('*.json'))]
    if stage=='fisher' and records:
     n_draws=len({r['draw'] for r in records})
     f=pd.DataFrame(records).drop(columns=['draw','sampled_token'])
     records=f.groupby(['layer','family']).mean().reset_index().to_dict('records')
     for r in records:r['draws']=n_draws
    for d in records:
     if d.get('status')=='zero_gradient':continue
     d=dict(d)
     if stage=='frontier':
      controls=d.pop('controls');d.pop('generated',None)
      d['repair']=float(d['answer_correct']) if not b['answer_correct'] else None
      d['regression']=float(not d['answer_correct']) if b['answer_correct'] else None
      d['kl']=np.mean([q['kl'] for q in controls]) if controls else None
      d['changed']=np.mean([q['prediction_changed'] for q in controls]) if controls else None
      valid=[q for q in controls if q['baseline_answer_correct'] and 'answer_correct' in q]
      d['control_regression']=np.mean([not q['answer_correct'] for q in valid]) if valid else None
      if d['channel'].startswith('random_'):d['seed']=d['channel'].split('_')[1];d['channel']='random'
     allrows[stage].append(dict(meta,**d))
 for model in root.glob('training_*'):
  for step in model.glob('step_*'):
   for path in (step/'facts').glob('*.json'):
    d=load(path);p=d['prompt']
    for r in d['rows']:
     r=dict(r);randoms=r.pop('random_projection_fractions')
     r['random_projection_fraction']=np.mean(randoms) if randoms else None
     r['projection_advantage']=r['projection_fraction']-np.mean(randoms) if randoms and r['projection_fraction'] is not None else None
     allrows['training'].append(dict(model=model.name,step=step.name,transition=d['transition'],
      **{k:p[k] for k in ['id','group_id','template','relation']},**r))
 pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
 definitions={
  'split':(['model','regime','layer','family'],['selection_fraction','content_selection_cosine','live_norm','content_norm','selection_norm','cross']),
  'fisher':(['model','regime','layer','family'],['live_sq','content_sq','selection_sq','cross']),
  'geometry':(['model','regime','layer'],['subject_norm','template_norm','first_norm','bias_norm','subject_exceeds_template']),
  'perturb':(['model','regime','layer','family','dose'],['live_change','frozen_change','selection_response','live_nonlinear_residual','abs_selection_response','abs_live_nonlinear_residual','relative_live_frozen_gap','frozen_affinity_error']),
  'frontier':(['model','regime','layer','family','channel','dose'],['margin_gain','answer_correct','repair','regression','kl','changed','control_regression']),
  'training':(['model','step','transition','layer','family','channel'],['projection_fraction','random_projection_fraction','projection_advantage','signed_delta_dot','delta_cosine'])}
 for stage,rows in allrows.items():
  if not rows:continue
  df=pd.DataFrame(rows);df.to_csv(out/(stage+'_raw.csv'),index=False)
  keys,metrics=definitions[stage];s=summarize(df,keys,metrics,cfg);s.to_csv(out/(stage+'_summary.csv'),index=False)
  plot(stage,s,out)
 dump(out/'interpretation.json',dict(
  correctness='Normalized accepted-answer prefix match on greedy generation; canonical first-token outcomes separate. Manually audit ambiguous aliases and repetition.',
  split='Selection includes attention, SiLU gates AND frozen normalization scales; global parameter gradients aggregate ALL token sites.',
  fisher='Monte Carlo conditional next-token model Fisher; content+selection+2cross=live. Margin outer products are not Fisher.',
  uncertainty='Equal-fact means; percentile subject-cluster bootstrap, no CI below five subjects; conditional on saved Monte Carlo draws. Not multiple-testing-corrected confirmatory inference.',
  geometry='Uncentered origin spans; approximate randomized top-r weight SVD. An origin norm crossing need not match a between-prompt variance crossing.',
  perturb='Single-matrix frozen response affine in that matrix. Live-minus-frozen is a defined finite response, not the integrated local selection gradient.',
  training='Retrospective endpoint correctness transitions on a prespecified cohort, not causal evidence about when a fact was learned. Net parameter displacement is not accumulated gradient covariance.',
  selection_fraction='||selection||/(||selection||+||content||), not an additive fraction of total sensitivity; cross can be negative.',
  curves='Plots are descriptive; choose dose/layer on separate calibration data before confirmatory evaluation.'))
 print('Reports saved to',out,flush=True)


def plot(stage,s,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for model,data in s.groupby('model'):
  fig,ax=plt.subplots(figsize=(10,6))
  if stage in ['split','fisher']:
   metric='selection_fraction' if stage=='split' else 'selection_sq'
   for (regime,family),g in data.groupby(['regime','family']):
    g=g.sort_values('layer');ax.plot(g.layer,g[metric],label=f'{regime}/{family}')
   ax.set(xlabel='Layer',ylabel=metric)
  elif stage=='geometry':
   for regime,g in data.groupby('regime'):
    g=g.sort_values('layer')
    for col in ['subject_norm','template_norm']:ax.plot(g.layer,g[col],label=f'{regime}/{col}')
   ax.set(xlabel='Layer',ylabel='Gate pre-activation origin norm')
  elif stage=='perturb':
   for (family,layer,regime),g in data.groupby(['family','layer','regime']):
    g=g.sort_values('dose');ax.plot(g.dose,g.abs_selection_response,label=f'{regime}/{family}/L{layer}')
   ax.set(xscale='log',xlabel='Relative Frobenius edit norm',ylabel='Absolute live minus frozen margin response')
  elif stage=='frontier':
   for (family,layer,channel,regime),g in data.groupby(['family','layer','channel','regime']):
    g=g.sort_values('dose');ax.plot(g.kl,g.margin_gain,'o-',label=f'{regime}/{family}/L{layer}/{channel}')
   ax.set(xlabel='Mean other-fact next-token KL (nats)',ylabel='Fixed-competitor margin gain')
  else:
   for (step,transition,family,channel),g in data.groupby(['step','transition','family','channel']):
    g=g.sort_values('layer');ax.plot(g.layer,g.projection_advantage,label=f'{step}/{transition}/{family}/{channel}')
   ax.set(xlabel='Layer',ylabel='Projection fraction minus random-subspace baseline')
  ax.set_title(f'{model}: {stage} (descriptive)');ax.grid(alpha=.2)
  if len(ax.lines)<=24:ax.legend(fontsize=6,loc='best')
  else:ax.text(.01,.99,'Many strata: see CSV for individual series',transform=ax.transAxes,va='top')
  fig.tight_layout();fig.savefig(out/f'{model}_{stage}.pdf');plt.close(fig)
