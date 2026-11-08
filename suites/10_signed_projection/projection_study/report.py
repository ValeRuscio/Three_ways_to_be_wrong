from pathlib import Path
import pandas as pd
from channel_study.io import dump,load
from channel_study.report import summarize


def report(root,cfg):
 root=Path(root);out=root/'reports';out.mkdir(parents=True,exist_ok=True);rows=[];paired=[];checks=[];baselines=[]
 for model in cfg['models']:
  for path in (root/model/'prompts').glob('*/measurement.json'):
   d=load(path);r=d['prompt'];meta=dict(model=model,id=str(r['id']),group_id=str(r['group_id']),relation=r['relation'],template=r['template'])
   rows.extend(dict(meta,**x) for x in d['rows']);paired.extend(dict(meta,**x) for x in d['paired']);checks.extend(dict(meta,**x) for x in d['finite_checks'])
   baselines.append(dict(meta,g=d['g'],c=d['c'],margin=d['native_margin'],baseline_error=d['baseline_error'],native_certificate=d['native_certificate']))
 if not rows:return
 df=pd.DataFrame(rows);df.to_csv(out/'projections_per_prompt.csv',index=False)
 metrics=['projection','cosine','gradient_norm','removed_projection','projection_over_content_norm','removed_over_content_norm','remaining_vs_all','total_selection_projection','content_gradient_norm']
 summary=summarize(df,['model','condition','layer'],metrics,cfg);summary.to_csv(out/'projections_summary.csv',index=False)
 contrasts=pd.DataFrame(paired);contrasts.to_csv(out/'contrasts_per_prompt.csv',index=False)
 cm=[x for x in contrasts.columns if x not in ['model','id','group_id','relation','template','layer']]
 summarize(contrasts,['model','layer'],cm,cfg).to_csv(out/'paired_contrasts_summary.csv',index=False)
 pd.DataFrame(baselines).to_csv(out/'coverage_and_baselines.csv',index=False)
 # Template-specific means make the independent-reference template choice visible.
 pd.concat([summarize(group,['model','condition','layer'],['projection','removed_projection'],cfg).assign(template=t) for t,group in df.groupby('template')],ignore_index=True).to_csv(out/'by_template_summary.csv',index=False)
 audit=dict(prompts=len(baselines),max_baseline_error=max(x['baseline_error'] for x in baselines),max_native_certificate=max(x['native_certificate'] for x in baselines))
 if checks:
  c=pd.DataFrame(checks);c.to_csv(out/'finite_difference_per_check.csv',index=False)
  c.groupby(['model','condition','layer','epsilon']).agg(checks=('passed','size'),failures=('passed',lambda x:int((~x).sum())),unresolved=('resolved',lambda x:int((~x).sum())),max_absolute_error=('absolute_error','max')).reset_index().to_csv(out/'finite_difference_summary.csv',index=False)
  audit.update(finite_checks=len(c),finite_failures=int((~c.passed).sum()),finite_unresolved=int((~c.resolved).sum()),max_finite_error=float(c.absolute_error.max()))
 dump(out/'audit.json',audit)
 dump(out/'interpretation.json',dict(
  signed_projection='p_S = <G_S, mu/||mu||>. G_S is the residual derivative at block entry, final prompt token; not a weight derivative.',
  effects='removed_projection=p_live-p_S. Positive means permitting that mechanism adds a positive mean-direction component in this singleton-freeze comparison.',
  normalization='Shared denominator ||G_all_frozen|| for normalized projections/effects. Unlike per-condition cosine, this preserves linear differences within each prompt.',
  contrasts='Norm-minus-gate and norm-minus-attention are PAIRED differences of removed projections. Positive does not imply an additive percentage allocation. singleton_sum_minus_total measures nonadditivity; nonzero is not an implementation error.',
  scope='Conditional derivative interventions preserve baseline predictions; not finite weight edits or demonstrations of factual error repair. Singleton effects can overlap. Comparisons at fixed reference mean, fixed objective and fixed native forward program.',
  uncertainty='Equal-fact averaging then subject-cluster bootstrap, conditional on these models and reference subjects. Includes no reference-sample uncertainty. Template0 reference means may influence cross-template comparisons; per-template summaries included. Unresolved finite checks are not strong positive certificates.'))
 plot(summary,out);print('Reports:',out,'Audit:',audit,flush=True)


def plot(df,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'live':'#111827','all_frozen':'#6b7280','attention_only':'#0d9488','gates_only':'#2563eb','scales_only':'#d97706'}
 for model,d in df.groupby('model'):
  fig,axs=plt.subplots(1,2,figsize=(12,4.5))
  for condition,v in d.groupby('condition'):
   v=v.sort_values('layer');axs[0].plot(v.layer,v.projection,label=condition,color=colors[condition]);axs[0].fill_between(v.layer,pd.to_numeric(v.projection_lo,errors='coerce'),pd.to_numeric(v.projection_hi,errors='coerce'),color=colors[condition],alpha=.1)
   if condition not in ['live','all_frozen']:
    axs[1].plot(v.layer,v.removed_over_content_norm,label=condition,color=colors[condition]);axs[1].fill_between(v.layer,pd.to_numeric(v.removed_over_content_norm_lo,errors='coerce'),pd.to_numeric(v.removed_over_content_norm_hi,errors='coerce'),color=colors[condition],alpha=.1)
  axs[0].set(title=model+': signed residual-mean projection',ylabel='Gradient dot unit reference mean')
  axs[1].set(title='Removed component: live minus singleton freeze',ylabel='Signed effect / norm of all-frozen gradient')
  for ax in axs:ax.axhline(0,color='gray',lw=.7);ax.set_xlabel('Layer (zero-indexed)');ax.legend(fontsize=8);ax.grid(alpha=.15)
  fig.tight_layout();fig.savefig(out/(model+'_signed_projection.pdf'));fig.savefig(out/(model+'_signed_projection.png'),dpi=150);plt.close(fig)
