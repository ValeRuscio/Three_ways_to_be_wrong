"""CPU-only reports. Unavailable estimates stay missing; no expected outcome is
hard-coded. P-values are exploratory; Holm families are explicit in output files.
"""
import json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .stats import boot,paired_p,auc_test,holm,trend_test
from .data import save


def write(rows,path):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 d.to_csv(path,index=False);return d

def adjusted(rows,path):
 d=pd.DataFrame(rows)
 if len(d) and 'p' in d:d['p_holm']=holm(pd.to_numeric(d.p,errors='coerce'))
 d.to_csv(path,index=False);return d

def estimate(v,g,cfg,median=False):return boot(v,g,cfg['bootstrap'],cfg['seed'],median)
def pvalue(v,g,cfg):return paired_p(v,g,cfg['permutations'],cfg['seed'])

def flatten(r,model,mode,spec):
 d={k:r.get(k) for k in ['id','group_id','relation','split','correct','eligible','exclusion','first_token_correct','canonical_first_correct','margin','tau','share','competitor_share','competitor_absolute_share','calibration_n','calibration_mode','handoff_cut','certificate']}
 d.update(model=model,mode=mode,family=spec['family'],stage=spec['stage'],step=spec['step'])
 for convention in ['full','hard20']:d['label_'+convention]=r.get('labels',{}).get(convention,'ineligible');d['q_'+convention]=r.get('q',{}).get(convention);d['q_max_'+convention]=r.get('q_max',{}).get(convention)
 return d

def origin_reports(collection,out,cfg):
 auc=[];controls=[];originrows=[];curves=[];handoff=[]
 for (key,mode),(spec,records) in collection.items():
  eligible=[r for r in records if r.get('eligible') and r['split']=='evaluation']; prefix=out/f'{key}_{mode}'
  sub=pd.DataFrame([flatten(r,key,mode,spec) for r in eligible])
  if len(sub):
   for metric in ['competitor_share','competitor_absolute_share']:
    d=sub[sub.label_full.isin(['selection','transport'])];result=auc_test((d.label_full=='selection').astype(int),d[metric],d.group_id,cfg['permutations'],cfg['seed'])
    auc.append(dict(model=key,mode=mode,metric=metric,**result))
  for r in records:
   if not r.get('eligible'):continue
   label='calibration_success' if r['split']=='calibration' and r['correct'] else r['labels']['full']
   if r['split']=='calibration' and (not r['correct'] or not r.get('calibration_reference',False)):continue
   for role,token in [('gold',r['g']),('competitor',r['c']),('decoy',r['replaced_competitor'])]:
    ledger=r['ledger'][str(token)]
    for origin,value in ledger['origins'].items():originrows.append(dict(model=key,mode=mode,id=r['id'],group_id=r['group_id'],split=r['split'],label=label,role=role,origin=origin,contribution=value))
   led=r['ledger'][str(r['g'])]
   for cut,value in enumerate(led['handoff']):curves.append(dict(model=key,mode=mode,id=r['id'],group_id=r['group_id'],label=label,cut=cut,depth=cut/(len(led['handoff'])-1),support=value))
   handoff.append(dict(model=key,mode=mode,id=r['id'],label=label,cut=r.get('handoff_cut'),eligible=r.get('handoff_cut') is not None))
  if eligible:
   for comparison in ['subject_minus_random_span','competitor_minus_decoy']:
    chosen=[r for r in eligible if comparison!='subject_minus_random_span' or r['random_span'] is not None]
    vals=[r['ledger'][str(r['c'])]['origins']['subject']-(r['ledger'][str(r['c'])]['random_span_support'] if comparison=='subject_minus_random_span' else r['ledger'][str(r['replaced_competitor'])]['origins']['subject']) for r in chosen]
    groups=[r['group_id'] for r in chosen];controls.append(dict(model=key,mode=mode,comparison=comparison,**estimate(vals,groups,cfg),p=pvalue(vals,groups,cfg)))
   fig,ax=plt.subplots(figsize=(5,4))
   for label in sorted({r['labels']['full'] for r in eligible}):
    rr=[r for r in eligible if r['labels']['full']==label]
    ax.scatter([r['tau'] for r in rr],[r['ledger'][str(r['c'])]['origins']['subject'] for r in rr],s=12,alpha=.5,label=label)
   ax.set(xlabel='Gold subject-origin contribution (logit units)',ylabel='Competitor subject-origin contribution');ax.legend();fig.tight_layout();fig.savefig(str(prefix)+'_origin_scatter.pdf');plt.close(fig)
 origins=write(originrows,out/'origins.csv');curve=write(curves,out/'handoff_curves.csv');write(handoff,out/'handoff_layers.csv')
 adjusted(auc,out/'origin_auc_holm.csv');adjusted(controls,out/'origin_controls_holm.csv')
 if len(origins):
  for (model,mode),d in origins[(origins.role=='competitor')&(origins.split=='evaluation')].groupby(['model','mode']):
   a=d.groupby(['label','origin']).contribution.mean().unstack().fillna(0);fig,ax=plt.subplots(figsize=(7,4));pos=np.zeros(len(a));neg=pos.copy()
   for name in ['subject','template','first_token','bias']:
    v=a[name].values;base=np.where(v>=0,pos,neg);ax.bar(a.index,v,bottom=base,label=name);pos+=np.maximum(v,0);neg+=np.minimum(v,0)
   ax.axhline(0,color='black',lw=.6);ax.set_ylabel('Mean signed competitor logit contribution');ax.legend();fig.tight_layout();fig.savefig(out/f'{model}_{mode}_origins.pdf');plt.close(fig)
 if len(curve):
  for (model,mode),d in curve.groupby(['model','mode']):
   fig,ax=plt.subplots(figsize=(6,4))
   for label,s in d.groupby('label'):
    a=s.groupby('depth').support.agg(median='median',lo=lambda x:x.quantile(.25),hi=lambda x:x.quantile(.75));ax.plot(a.index,a['median'],label=label);ax.fill_between(a.index,a.lo,a.hi,alpha=.15)
   ax.set(xlabel='Normalized cut depth',ylabel='Subject-origin gold support at answer position');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/f'{model}_{mode}_handoff.pdf');plt.close(fig)
 h=pd.DataFrame(handoff)
 if len(h):
  for (model,mode),d in h.groupby(['model','mode']):
   fig,ax=plt.subplots(figsize=(6,4))
   for label in ['calibration_success','correct','selection']:
    a=d[(d.label==label)&d.eligible]['cut'].dropna()
    if len(a):ax.hist(a,bins=np.arange(int(h['cut'].max())+2)-.5,alpha=.4,label=f'{label} (n={len(a)})')
   ax.set(xlabel='First half-support cut (eligible cases only)',ylabel='Count');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/f'{model}_{mode}_handoff_distribution.pdf');plt.close(fig)


def alignment_reports(df,out,cfg,collection):
 transitions=[];cells=[];effects=[];odds=[];agreement=[]
 for (family,mode),s in df[(df.split=='evaluation')&df.step.isna()].groupby(['family','mode']):
  stages=[v for v in ['base','sft','dpo','instruct'] if v in set(s.stage)]
  pairs=list(zip(stages[:-1],stages[1:]))
  if len(stages)>2:pairs.append(('base','instruct'))
  for left,right in pairs:
   a=s[s.stage==left];b=s[s.stage==right];m=a.merge(b,on=['id','group_id','relation'],suffixes=('_a','_b'),validate='one_to_one');name=f'{family}:{left}->{right}'
   if not len(m):continue
   m['cell']=np.select([~m.correct_a&~m.correct_b,~m.correct_a&m.correct_b,m.correct_a&~m.correct_b],['fails_both','fixed','broken'],default='correct_both')
   for cell,ss in m.groupby('cell'):cells.append(dict(pair=name,mode=mode,cell=cell,n=len(ss)))
   for convention in ['full','hard20']:
    la='label_'+convention+'_a';lb='label_'+convention+'_b'
    for (cell,x,y),ss in m.groupby(['cell',la,lb]):transitions.append(dict(pair=name,mode=mode,convention=convention,cell=cell,base_label=x,aligned_label=y,n=len(ss)))
    for variable in ['q_'+convention,'q_max_'+convention,'tau','margin']:
     ok=m[[variable+'_a',variable+'_b']].notna().all(axis=1);ss=m[ok];delta=ss[variable+'_b']-ss[variable+'_a']
     effects.append(dict(pair=name,mode=mode,convention=convention,variable=variable,**estimate(delta,ss.group_id,cfg),p=pvalue(delta,ss.group_id,cfg)))
    f=m[(~m.correct_a)&m[la].isin(['source','transport','selection'])]
    for label in ['source','transport','selection']:
     def logor(t):
      selected=t[la]==label;fixed=t.correct_b;counts=[sum(selected&fixed),sum(selected&~fixed),sum(~selected&fixed),sum(~selected&~fixed)]
      a,b,c,d=np.array(counts)+.5;return float(np.log(a*d/(b*c)))
     if not len(f):continue
     rng=np.random.default_rng(cfg['seed']);groups=f.group_id.unique();blocks=[f[f.group_id==g] for g in groups];vals=[];perms=[];observed=logor(f)
     for _ in range(cfg['bootstrap']):vals.append(logor(pd.concat([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])))
     # Cluster sign-permutation is not valid for OR; use cluster bootstrap CI,
     # and a subject-level label permutation only for one-row-per-subject panels.
     p=None
     if f.group_id.nunique()==len(f):
      for _ in range(cfg['permutations']):
       z=f.copy();z[la]=rng.permutation(z[la].values);perms.append(logor(z))
      p=float((1+sum(abs(v)>=abs(observed) for v in perms))/(len(perms)+1))
     odds.append(dict(pair=name,mode=mode,convention=convention,label=label,odds_ratio=np.exp(observed),lo=np.exp(np.quantile(vals,.025)),hi=np.exp(np.quantile(vals,.975)),n=len(f),p=p))
   both=m[m.cell=='fails_both'];agreement.append(dict(pair=name,mode=mode,n=len(both),transition_agreement=float(((both.label_full_a==both.label_hard20_a)&(both.label_full_b==both.label_hard20_b)).mean()) if len(both) else None))
 write(cells,out/'alignment_cells.csv');write(transitions,out/'alignment_transitions.csv');adjusted(effects,out/'alignment_paired_effects_holm.csv');adjusted(odds,out/'alignment_fix_odds_holm.csv');write(agreement,out/'alignment_decoy_agreement.csv')


def training_reports(df,out,cfg):
 trajectories=[];summary=[];trends=[];last_fail=[]
 for (family,mode),s in df[(df.split=='evaluation')&df.step.notna()].groupby(['family','mode']):
  s=s.copy();s['step']=s.step.astype(int)
  for (model,step),d in s.groupby(['model','step']):
   failures=d[~d.correct];known=failures[failures.label_full.isin(['source','transport','selection'])]
   row=dict(family=family,mode=mode,model=model,step=step,n=len(d),accuracy=d.correct.mean(),first_token_accuracy=d.first_token_correct.mean(),first_token_n=int(d.first_token_correct.notna().sum()),
     first_token_failures=int((d.first_token_correct==False).sum()),full_answer_failures=len(failures),labelled_failures=len(known),calibration_n_min=d.calibration_n.min(),
     source_count=sum(known.label_full=='source'),label_coverage=len(known)/max(1,len(failures)))
   row.update({label:float((known.label_full==label).mean()) if len(known) else None for label in ['source','transport','selection']});summary.append(row)
  for identifier,d in s.groupby('id'):
   d=d.sort_values('step');good=d[d.correct]
   first=int(good.step.iloc[0]) if len(good) else None
   previous=d[(d.step<first)&~d.correct] if first is not None else d.iloc[0:0]
   last_fail.append(dict(family=family,mode=mode,id=identifier,group_id=d.group_id.iloc[0],first_correct_step=first,last_failure_label=previous.label_full.iloc[-1] if len(previous) else None,
      later_regression=bool((~d[d.step>first].correct).any()) if first is not None else False))
   for _,r in d.iterrows():trajectories.append(dict(family=family,mode=mode,id=identifier,step=r.step,label=r.label_full,first_correct_step=first))
  a=pd.DataFrame([r for r in summary if r['family']==family and r['mode']==mode]).sort_values('step')
  stat=trend_test(a.source_count,a.labelled_failures,a.step)
  # Cluster bootstrap of slope of conditional source share over log training step.
  rng=np.random.default_rng(cfg['seed']);groups=s.group_id.unique();blocks=[s[s.group_id==g] for g in groups]
  def slope(t):
   q=t[t.label_full.isin(['source','transport','selection'])].assign(source=lambda z:z.label_full=='source').groupby('step').source.mean()
   return float(np.polyfit(np.log(q.index),q.values,1)[0]) if len(q)>1 else np.nan
  values=[slope(pd.concat([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])) for _ in range(cfg['bootstrap'])];values=np.array(values);values=values[np.isfinite(values)]
  trends.append(dict(family=family,mode=mode,**stat,slope=slope(s),slope_lo=np.quantile(values,.025) if len(values) else None,slope_hi=np.quantile(values,.975) if len(values) else None,note='CA p assumes independence and is descriptive only; use subject-cluster slope interval'))
  fig,ax=plt.subplots(figsize=(7,4))
  for label in ['source','transport','selection']:ax.plot(a.step,a[label],marker='o',label=label)
  ax.set_xscale('log');ax.set(xlabel='Training step',ylabel='Share among labelled failures');other=ax.twinx();other.plot(a.step,a.accuracy,'k--',label='accuracy');other.set_ylabel('Full fixed-set accuracy');ax.legend();fig.tight_layout();fig.savefig(out/f'{family}_{mode}_training.pdf');plt.close(fig)
  panel=s.pivot(index='id',columns='step',values='label_full');order=s.groupby('id').apply(lambda z:z.loc[z.correct,'step'].min(),include_groups=False).sort_values(na_position='last').index;panel=panel.reindex(order)
  colors={'source':0,'transport':1,'selection':2,'correct':3,'undetermined':4,'ineligible':5};arr=panel.map(lambda x:colors.get(x,5)).values
  fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(arr,aspect='auto',vmin=0,vmax=5,cmap='tab10',interpolation='nearest');ax.set_xticks(range(len(panel.columns)),panel.columns,rotation=45);ax.set(xlabel='Training step',ylabel='Fixed prompts, ordered by first success');bar=fig.colorbar(im,ax=ax,ticks=range(6));bar.ax.set_yticklabels(list(colors));fig.tight_layout();fig.savefig(out/f'{family}_{mode}_trajectories.pdf');plt.close(fig)
 write(summary,out/'training_summary.csv');write(trajectories,out/'training_trajectories.csv');write(trends,out/'training_trends.csv');lf=write(last_fail,out/'last_failure.csv')
 if cfg.get('exposure_csv') and len(lf):
  exposure=pd.read_csv(cfg['exposure_csv'],dtype={'id':str})
  if not {'id','exposure'}.issubset(exposure):raise ValueError('Exposure CSV requires id,exposure; actual corpus counts, not popularity')
  if exposure.id.duplicated().any():raise ValueError('Duplicate exposure IDs')
  merged=lf.merge(exposure,on='id',how='left',validate='many_to_one');merged['exposure_bin']=pd.qcut(merged.exposure.rank(method='dense'),3,duplicates='drop');write(merged,out/'last_failure_by_exposure.csv')
  exposure_summary=merged.groupby(['family','mode','exposure_bin'],observed=True).agg(n=('id','size'),observed_first_success=('first_correct_step','count'),median_first_success_step=('first_correct_step','median')).reset_index()
  write(exposure_summary,out/'exposure_summary.csv')
  distribution=merged.groupby(['family','mode','exposure_bin','last_failure_label'],observed=True).size().rename('n').reset_index();write(distribution,out/'exposure_last_failure_counts.csv')
 else:save(out/'exposure_status.json',{'status':'not_run','reason':'Provide actual corpus co-occurrence counts via exposure_csv; PopQA popularity is not exposure.'})


def repair_reports(collection,out,cfg):
 rows=[];coverage=[];plants=[]
 for (key,mode),(spec,records) in collection.items():
  directory=out.parent/key/mode
  for path in (directory/'repair_records').glob('*.json'):
   obj=json.loads(path.read_text());coverage.append(dict(model=key,mode=mode,id=obj['id'],donor=obj['donor_available'],wrong_donor=obj['wrong_donor_available']))
   rows.extend(dict(model=key,mode=mode,**r) for r in obj['rows'])
  for path in (directory/'plant_records').glob('*.json'):
   obj=json.loads(path.read_text());plants.extend(dict(model=key,mode=mode,**r) for r in obj['audit']);rows.extend(dict(model=key,mode=mode,**r) for r in obj['rows'])
 write(coverage,out/'donor_coverage.csv');write(plants,out/'planted_controls.csv');df=write(rows,out/'repairs.csv')
 if not len(df):return
 df['layer_key']=df.layer.fillna(-1);keys=['model','mode','plant','label','correct_before','family','arm','layer_key','dose']
 curves=[];minimum=[]
 for key,d in df.groupby(keys,dropna=False):curves.append(dict(zip(keys,key),**estimate(d.margin_gain,d.group_id,cfg,True),recovery_rate=d.recovered.mean()))
 write(curves,out/'dose_response.csv')
 for key,d in df.groupby(['model','mode','plant','id','group_id','label','family','arm','layer_key'],dropna=False):
  good=d[d.recovered];minimum.append(dict(zip(['model','mode','plant','id','group_id','label','family','arm','layer_key'],key),minimum_tested_dose=float(good.dose.min()) if len(good) else None,right_censored=not len(good),max_tested_dose=d.dose.max()))
 write(minimum,out/'minimum_tested_dose.csv')
 comparisons=[]
 for family,active,control in [('source','donor','random'),('source','donor','wrong_subject'),('transport','top_heads','random_span'),('transport','top_heads','bottom_heads'),('selection','answer_up','random'),('selection','alternative_down','random'),('transport_secondary','donor_provenance','random'),('selection_upstream','competitor_projection','random')]:
  d=df[df.family==family];idx=['model','mode','plant','id','group_id','label','correct_before','layer_key','dose']
  if not len(d):continue
  pivot=d.pivot(index=idx,columns='arm',values='margin_gain')
  if active not in pivot or control not in pivot:continue
  p=pivot[[active,control]].dropna().reset_index();p['difference']=p[active]-p[control]
  group=['model','mode','plant','label','correct_before','layer_key','dose']
  for key,ss in p.groupby(group,dropna=False):comparisons.append(dict(zip(group,key),family=family,active=active,control=control,**estimate(ss.difference,ss.group_id,cfg,True),p=pvalue(ss.difference,ss.group_id,cfg)))
 adjusted(comparisons,out/'repair_control_comparisons_holm.csv')
 # Predeclared 3x3 slice, not best-on-test doses. Full grids always exported.
 main=df[(df.plant=='natural')&~df.correct_before&(((df.family=='source')&(df.arm=='donor')&(df.dose==1.))|((df.family=='transport')&(df.arm=='top_heads')&(df.dose==2.))|((df.family=='selection')&(df.arm=='alternative_down')&(df.dose==1.)))]
 # Primary source layer is the median of the predeclared layer grid.
 keep=[]
 for (model,mode),sub in main.groupby(['model','mode']):
  layers=sorted(sub.loc[sub.family=='source','layer_key'].unique())
  middle=layers[len(layers)//2] if layers else None
  keep.append(sub[(sub.family!='source')|(sub.layer_key==middle)])
 main=pd.concat(keep) if keep else main
 cells=[]
 for key,d in main.groupby(['model','mode','label','family','layer_key']):cells.append(dict(zip(['model','mode','label','family','layer_key'],key),**estimate(d.margin_gain,d.group_id,cfg,True)))
 write(cells,out/'repair_matrix_declared_slice.csv')
 nine=[]
 control_choice={'source':'random','transport':'random_span','selection':'random'}
 for (model,mode),sub in main.groupby(['model','mode']):
  for (label,family),active in sub.groupby(['label','family']):
   if label not in ['source','transport','selection']:continue
   control=df[(df.model==model)&(df['mode']==mode)&(df.plant=='natural')&(df.family==family)&(df.arm==control_choice[family])]
   paired=active.merge(control,on=['id','group_id','dose','layer_key'],suffixes=('_a','_c'))
   delta=paired.margin_gain_a-paired.margin_gain_c
   nine.append(dict(model=model,mode=mode,label=label,family=family,**estimate(delta,paired.group_id,cfg,True),p=pvalue(delta,paired.group_id,cfg)))
 nine_df=pd.DataFrame(nine)
 if len(nine_df):
  nine_df['p_holm_nine']=nine_df.groupby(['model','mode']).p.transform(lambda p:holm(pd.to_numeric(p,errors='coerce')))
 write(nine_df,out/'repair_matrix_nine_control_tests.csv')
 if 'final_normalized_relative_cost' in df:
  cost_comparisons(df,out,cfg)
 for (model,mode,plant),d in df.groupby(['model','mode','plant']):
  fig,axes=plt.subplots(1,3,figsize=(13,4))
  for ax,family,arm in zip(axes,['source','transport','selection'],['donor','top_heads','alternative_down']):
   sub=d[(d.family==family)&(d.arm==arm)&~d.correct_before]
   for (label,layer),ss in sub.groupby(['label','layer_key']):
    a=ss.groupby('dose').margin_gain.median();ax.plot(a.index,a.values,marker='.',label=f'{label}, layer {layer:g}')
   ax.set(title=family,xlabel='Native intervention dose',ylabel='Median margin gain');ax.legend(fontsize=5)
  fig.tight_layout();fig.savefig(out/f'{model}_{mode}_{plant}_dose_response.pdf');plt.close(fig)


def cost_comparisons(df,out,cfg):
 # Cost is a downstream disturbance proxy, not a claim that intervention types
 # are causally equivalent. No extrapolation and no optimal-dose selection.
 candidates=df[(df.plant=='natural')&~df.correct_before&(((df.family=='source')&(df.arm=='donor'))|((df.family=='transport')&(df.arm=='top_heads'))|((df.family=='selection')&(df.arm=='alternative_down')))].copy()
 values=[]
 for key,d in candidates.groupby(['model','mode','id','group_id','label','family']):
  if key[-1]=='source':
   layers=sorted(d.layer_key.unique());d=d[d.layer_key==layers[len(layers)//2]]
  x=d.groupby('final_normalized_relative_cost').margin_gain.mean().sort_index()
  for cost in cfg['common_cost_grid']:
   if len(x)>=2 and x.index.min()<=cost<=x.index.max():values.append(dict(zip(['model','mode','id','group_id','label','family'],key),cost=cost,gain=float(np.interp(cost,x.index,x.values))))
 frame=write(values,out/'repairs_common_cost_interpolation.csv');rows=[];pooled=[]
 if not len(frame):return
 for (mode,cost,label),d in frame.groupby(['mode','cost','label']):
  if label not in ['source','transport','selection']:continue
  pivot=d.pivot(index=['model','id','group_id'],columns='family',values='gain')
  for other in ['source','transport','selection']:
   if other==label or other not in pivot or label not in pivot:continue
   paired=pivot[[label,other]].dropna().reset_index();paired['difference']=paired[label]-paired[other]
   for model,ss in paired.groupby('model'):rows.append(dict(mode=mode,cost=cost,label=label,comparison=f'{label} minus {other}',model=model,**estimate(ss.difference,ss.group_id,cfg),p=pvalue(ss.difference,ss.group_id,cfg)))
   if len(paired)>2 and paired.group_id.nunique()>1:
    from .stats import fixed_effects
    result=fixed_effects(paired.difference,paired.model,paired.group_id)
    pooled.append(dict(mode=mode,cost=cost,label=label,comparison=f'{label} minus {other}',**result,n=len(paired),note='Sum-coded model fixed effects; subject-cluster covariance with t interval; common downstream disturbance proxy'))
 adjusted(rows,out/'repair_cross_family_common_cost_holm.csv');adjusted(pooled,out/'repair_pooled_fixed_effects_holm.csv')


def report(root,cfg):
 root=Path(root);out=root/'reports';out.mkdir(exist_ok=True);collection={};rows=[];coverage=[];audit=[];certificates=[]
 for path in sorted(root.glob('*/*/measurements.json')):
  spec=json.loads((path.parent/'model.json').read_text());records=json.loads(path.read_text());key=spec['key'];mode=path.parent.name;collection[(key,mode)]=(spec,records)
  rows.extend(flatten(r,key,mode,spec) for r in records)
  for r in records:
   audit.append(dict(model=key,mode=mode,**{k:r.get(k) for k in ['id','group_id','split','relation','subject','prompt','answer','generated','correct','first_token_correct','eligible','exclusion']}))
   for role,ledger in r.get('ledger',{}).items():certificates.append(dict(model=key,mode=mode,id=r['id'],token=role,reconstruction_error=ledger['reconstruction_error'],handoff_error=ledger['handoff_error']))
  for split in ['calibration','evaluation']:
   rr=[r for r in records if r['split']==split];coverage.append(dict(model=key,mode=mode,split=split,n=len(rr),correct=sum(r['correct'] for r in rr),eligible=sum(r.get('eligible',False) for r in rr),accuracy=sum(r['correct'] for r in rr)/max(1,len(rr))))
 df=write(rows,out/'measurements.csv');write(coverage,out/'coverage.csv');write(audit,out/'annotation_audit.csv');write(certificates,out/'certificates.csv')
 if not len(df):raise RuntimeError('No measurements found')
 origin_reports(collection,out,cfg);alignment_reports(df,out,cfg,collection);training_reports(df,out,cfg);repair_reports(collection,out,cfg)
 save(out/'interpretation.json',dict(origin='Exact realized affine accounting, not causal identification. Signed shares can be negative or exceed one. Template group includes all remaining prompt positions, including chat markers.',
  handoff='First half-support crossing conditional on positive success-calibrated floor; nonmonotone curves retained. Calibration successes are not evaluation accuracy.',
  repairs='Every intervention includes all-vocabulary effects. No diagonal dominance assumed. Native alpha/gamma/output-state doses have incompatible units; no native cross-family ratio is reported.',
  controls='Random positions and decoys are hypotheses, not presumed zero. Success interventions may change margins substantially. Hard decoys selected once by first family member and reused.',
  inference='Holm correction across rows in each *_holm.csv. Bootstrap is subject-clustered within model. Training naive trend p is descriptive, not independent-sample evidence.',
  limitations='One prompt format per primary condition; canonical first-token diagnostics only. Full-answer and first-token accuracy include all fixed prompts. Verify generation audit before publication.'))
 print(f'Reports saved to {out}',flush=True)
