"""Read-only post-hoc regime joins; do not alter the previous campaign protocol."""
from pathlib import Path
import numpy as np
import pandas as pd
from channel_study.io import load,dump
from channel_study.report import summarize
from .regimes import calibrate,classify


def write(df,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');df.to_csv(tmp,index=False);tmp.replace(path)


def summary(rows,keys,metrics,cfg):
 if not rows:return pd.DataFrame()
 return summarize(pd.DataFrame(rows),keys,metrics,cfg)


def report(root,cfg):
 root=Path(root);out=root/'reports';out.mkdir(exist_ok=True)
 collected={};allregimes=[];allcal=[];allcoverage=[];var=[];sink=[]
 for spec in cfg['models']:
  model=spec['key'];directory=root/model;records=[];folders=[]
  for folder in sorted((directory/'prompts').glob('*')):
   if not (folder/'baseline.json').exists():continue
   b=load(folder/'baseline.json');b['folder']=folder
   if (folder/'origins.json').exists():b['origins']=load(folder/'origins.json')
   records.append(b)
  calibration=[r for r in records if r['prompt']['split']=='calibration' and 'origins' in r]
  thresholds={q:calibrate(calibration,q,cfg['min_calibration_subjects']) for q in cfg['source_quantiles']}
  dump(directory/'calibration.json',{str(q):v for q,v in thresholds.items()})
  for q,refs in thresholds.items():
   allcal.extend(dict(model=model,relation=rel,**v) for rel,v in refs.items())
  batches={k:[] for k in ['profile','origins','null','freeze','certificate','matrix']}
  for b in records:
   p=b['prompt'];folder=b['folder']
   if p['split']!='evaluation':continue
   regime=classify(b,thresholds[cfg['primary_quantile']]) if 'origins' in b else ('answer_success' if b['answer_correct'] else 'unlabelled_answer_failure')
   meta=dict(model=model,id=p['id'],group_id=p['group_id'],relation=p['relation'],template=p['template'],regime=regime)
   coverage=dict(meta,answer_correct=b['answer_correct'],canonical_top1=b['canonical_top1'],first_token_id=b.get('first_token_id'),first_token_in_subject=b.get('first_token_in_subject'),tokenizer_bos_id=b.get('tokenizer_bos_id'));allcoverage.append(coverage)
   if 'origins' in b:
    o=b['origins'];measure={}
    for group,values in [('gold',o['centered_gold']),('competitor',o['centered_competitor']),('margin',o['margin_origins'])]:
     measure.update({group+'_'+k:v for k,v in values.items()})
    measure.update(gold_source_share=o['gold_source_share'],competitor_source_share=o['competitor_source_share'],origin_certificate=o['certificate'],gate_certificate=o['gate_certificate'])
    for q,refs in thresholds.items():allregimes.append(dict(meta,quantile=q,regime_at_quantile=classify(b,refs),**measure))
    batches['origins'].append(dict(meta,**measure))
   if (folder/'split.json').exists():
    native=load(folder/'split.json');batches['profile'].extend(dict(meta,**r) for r in native)
   else:native=[]
   if (folder/'null.json').exists():
    batches['null'].extend(dict(meta,**r) for r in load(folder/'null.json'))
    batches['null'].extend(dict(meta,condition='trained_margin',seed=0,**r) for r in native)
   if (folder/'random_init.json').exists():batches['null'].extend(dict(meta,condition='random_initialization',seed=cfg['random_initialization_seed'],**r) for r in load(folder/'random_init.json')['rows'])
   for stage in ['freeze','certificate','matrix']:
    if (folder/(stage+'.json')).exists():batches[stage].extend(dict(meta,**r) for r in load(folder/(stage+'.json'))['rows'])
  definitions={
   'profile':(['model','regime','layer','family'],['selection_fraction','content_selection_cosine','content_norm','selection_norm','live_norm','cross']),
   'origins':(['model','regime'],['gold_subject','gold_template','gold_first','gold_bias','competitor_subject','margin_subject','margin_template','margin_first','margin_bias','gold_source_share','competitor_source_share','origin_certificate','gate_certificate']),
   'null':(['model','regime','condition','layer','family'],['selection_fraction','content_selection_cosine','content_norm','selection_norm','live_norm']),
   'freeze':(['model','regime','layer','family'],['attention_norm','gate_norm','norm_norm','attention_signed_projection_share','gate_signed_projection_share','norm_signed_projection_share','attention_one_freeze_norm','gate_one_freeze_norm','norm_one_freeze_norm','attention_gate_cross','attention_norm_cross','gate_norm_cross','closure']),
   'certificate':(['model','layer','family','direction','epsilon'],['live_absolute_error','live_pass','live_resolved','frozen_absolute_error','frozen_pass']),
   'matrix':(['model','regime','source_layer','frozen_gate_layer'],['effect_norm','selection_norm','relative_effect','signed_selection_alignment'])}
  if cfg.get('freeze_mode')=='singletons':definitions['freeze']=(['model','regime','condition','layer','family'],['selection_fraction','content_selection_cosine','content_norm','selection_norm','live_norm','cross'])
  for stage,rows in batches.items():
   if not rows:continue
   keys,metrics=definitions[stage];df=summary(rows,keys,metrics,cfg);collected.setdefault(stage,[]).append(df)
   if stage=='certificate':
    raw=pd.DataFrame(rows);group=['model','layer','family','direction','epsilon']
    extrema=raw.groupby(group)[['live_absolute_error','frozen_absolute_error']].max().add_suffix('_max').reset_index()
    df=df.merge(extrema,on=group);collected[stage][-1]=df
    # Independent maximum errors, failures, and unresolved checks remain visible.
    dump(directory/'certificate_status.json',dict(checks=len(rows),live_failures=int((~raw.live_pass).sum()),
     frozen_failures=int((~raw.frozen_pass).sum()),unresolved=int((~raw.live_resolved).sum()),max_live_error=float(raw.live_absolute_error.max()),max_frozen_error=float(raw.frozen_absolute_error.max())))
  # Paired null differences, same prompt / matrix. No independent-sample CI shortcut.
  if batches['null']:
   df=pd.DataFrame(batches['null']);group=['model','id','group_id','template','layer','family','condition']
   metrics=['selection_fraction','content_selection_cosine'];df=df.groupby(group)[metrics].mean().reset_index()
   reference=df[df.condition=='trained_margin'].drop(columns='condition')
   comparisons=[]
   for condition in ['random_readout','random_initialization']:
    paired=reference.merge(df[df.condition==condition].drop(columns='condition'),on=[k for k in group if k!='condition'],suffixes=('_trained','_null'))
    for metric in metrics:paired[metric+'_difference']=paired[metric+'_trained']-paired[metric+'_null']
    paired['condition']=condition;comparisons.append(paired)
   if comparisons:
    pairs=pd.concat(comparisons,ignore_index=True)
    ans=summary(pairs.to_dict('records'),['model','condition','layer','family'],[m+'_difference' for m in metrics],cfg)
    collected.setdefault('null_paired',[]).append(ans)
  for path in (directory/'variance').glob('*.json'):
   d=load(path);var.extend(dict(model=model,relation=d['relation'],**r) for r in d['rows'])
  for path in (directory/'sink').glob('*.json'):
   d=load(path);sink.extend(dict(model=model,relation=d['relation'],template=cfg['variance_templates'][r['template_slot']],**r) for r in d['rows'])
 for stage,frames in collected.items():
  df=pd.concat(frames,ignore_index=True);write(df,out/(stage+'_summary.csv'));plot(stage,df,out)
 if allregimes:
  df=pd.DataFrame(allregimes);write(df,out/'regime_sensitivity.csv')
  primary=df[np.isclose(df['quantile'],cfg['primary_quantile'])].copy();primary['regime']=primary['regime_at_quantile']
  write(primary,out/'regimes.csv');write(primary.groupby(['model','regime']).agg(prompts=('id','size'),facts=('id','nunique'),subjects=('group_id','nunique')).reset_index(),out/'regime_counts.csv')
 if allcal:write(pd.DataFrame(allcal),out/'calibration.csv')
 if allcoverage:write(pd.DataFrame(allcoverage),out/'coverage.csv')
 if var:
  df=pd.DataFrame(var);write(df,out/'variance_summary.csv')
  # Pool sums of squares within relations; NOT an across-relation ANOVA.
  pooled=df.groupby(['model','measurement','layer','site'])[['subject_ss','template_ss','interaction_ss','total_ss']].sum().reset_index()
  for component in ['subject','template','interaction']:pooled[component+'_fraction']=pooled[component+'_ss']/pooled.total_ss.replace(0,np.nan)
  write(pooled,out/'variance_pooled.csv');cross=[]
  for keys,v in pooled.groupby(['model','measurement','site']):
   v=v.sort_values('layer');diff=v.subject_ss-v.template_ss
   crosses=v.layer[(diff>0)&(diff.shift(1)<=0)].tolist()
   cross.append(dict(model=keys[0],measurement=keys[1],site=int(keys[2]),crossings=crosses))
  dump(out/'variance_crossings.json',cross)
 if sink:write(pd.DataFrame(sink),out/'sink_summary.csv')
 dump(out/'interpretation.json',dict(
  regimes='Vocabulary-centered gold/competitor origin diagnostics, calibrated on independent canonical-token + accepted-answer successes. Not the earlier labels unchanged, and not proven causal stages. Continuation/mixed/uncalibrated cases retained.',
  freeze=('Singleton freeze sensitivity: overlapping marginal effects, not additive attribution.' if cfg.get('freeze_mode')=='singletons' else 'Exact Shapley attribution of the three-group freeze derivative game.'),
  norms='Includes all RMSNorm/LayerNorm modules, including OLMo Q/K normalization and final norm.',
  variance='Balanced subject x templates 0/1/2; interaction retained. Attention decomposes semantic role masses, not incomparable raw token slots.',
  sink='Answer-site pre-gate first-token origin vector constancy conditional on relation and template. Low variance alone does not prove an output unigram prior or causal bias function.',
  matrix='One-downstream-layer gate freeze per column. Marginal effects overlap; no additive column sum or unique causal-stage assignment.',
  certificates='Centered live finite differences over epsilon sweep plus single-matrix frozen finite-response affinity. Report failures and unresolved directions, not just average error.',
  bias='Family comparisons are observational. QKV bias presence is confounded with training, normalization and architecture; no causal attribution to biases.',
  bootstrap='Equal-fact averages and subject-cluster bootstrap; no confidence interval with fewer than five subjects; exploratory, not multiplicity-corrected.'))
 print('Reports saved:',out,flush=True)


def plot(stage,df,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 if stage not in ['profile','null_paired','freeze','matrix']:return
 for model,group in df.groupby('model'):
  if stage=='matrix':
   regimes=list(group.regime.unique());fig,axs=plt.subplots(1,len(regimes),figsize=(5*len(regimes),4),squeeze=False)
   for ax,regime in zip(axs[0],regimes):
    z=group[group.regime==regime].pivot(index='source_layer',columns='frozen_gate_layer',values='signed_selection_alignment')
    z=z.mask(z.columns.to_numpy()[None,:]<=z.index.to_numpy()[:,None])
    im=ax.imshow(z,aspect='auto',origin='lower',cmap='coolwarm');fig.colorbar(im,ax=ax);ax.set(title=regime,xlabel='Frozen gate layer',ylabel='Weight source layer')
  else:
   fig,ax=plt.subplots(figsize=(10,6))
   family='down';subset=group[group.family==family]
   if stage=='profile':
    for regime,v in subset.groupby('regime'):ax.plot(v.layer,v.selection_fraction,label=regime)
    ax.set_ylabel('Selection norm fraction (down projection)')
   elif stage=='null_paired':
    for condition,v in subset.groupby('condition'):ax.plot(v.layer,v.selection_fraction_difference,label=condition)
    ax.axhline(0,color='black',lw=.5);ax.set_ylabel('Trained margin minus matched null (down projection)')
   elif 'condition' in subset.columns:
    for (regime,condition),v in subset.groupby(['regime','condition']):ax.plot(v.layer,v.selection_norm,label=regime+'/'+condition)
    ax.set_ylabel('Live minus conditional-gradient norm (overlapping effects)')
   else:
    for regime,v in subset.groupby('regime'):
     for mechanism in ['attention','gate','norm']:ax.plot(v.layer,v[mechanism+'_signed_projection_share'],label=regime+'/'+mechanism)
    ax.set_ylabel('Signed Shapley projection share (down projection)')
   ax.set_xlabel('Layer');ax.legend(fontsize=6);ax.grid(alpha=.2)
  fig.suptitle(model+' '+stage);fig.tight_layout();fig.savefig(out/(model+'_'+stage+'.pdf'));plt.close(fig)
