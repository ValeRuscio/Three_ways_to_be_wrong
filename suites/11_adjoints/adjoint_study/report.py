from pathlib import Path
import json
import numpy as np
import pandas as pd
from channel_study.io import dump

METRICS=['angle_degrees','live_frozen_cosine','frozen_over_live','selection_over_live','selection_fraction','content_selection_cosine']
META=['model','prompt_key','id','group_id','relation','template','regime']


def summarize(df,keys,metrics,nboot=1000):
    """Equal-fact means; subject-cluster resampling, conditional on fixed labels."""
    out=[];rng=np.random.default_rng(113)
    for values,a in df.groupby(keys,dropna=False):
        if not isinstance(values,tuple):values=(values,)
        row=dict(zip(keys,values));f=a.groupby(['group_id','id'])[metrics].mean().reset_index()
        row.update(prompts=a.prompt_key.nunique(),facts=len(f),subjects=f.group_id.nunique())
        for m in metrics:
            valid=f[['group_id',m]].dropna();row[m]=float(valid[m].mean()) if len(valid) else None
            row[m+'_valid_facts']=len(valid);row[m+'_lo']=row[m+'_hi']=None
            ids=valid.group_id.unique()
            if len(ids)>=5:
                grouped=valid.groupby('group_id')[m].agg(['sum','count']).reindex(ids)
                draws=rng.integers(len(ids),size=(nboot,len(ids)))
                boot=grouped['sum'].to_numpy()[draws].sum(1)/grouped['count'].to_numpy()[draws].sum(1)
                row[m+'_lo'],row[m+'_hi']=map(float,np.quantile(boot,[.025,.975]))
        out.append(row)
    return pd.DataFrame(out)


def relation_contrasts(df,nboot):
    """Within-relation regime-success contrasts. No imputation of uncrossed depth."""
    out=[];rng=np.random.default_rng(552)
    for (model,relation),a in df.groupby(['model','relation']):
        for target in ['absent_origin','overridden_origin','misretrieved_origin']:
            sub=a[a.regime.isin(['answer_success',target])]
            for metric in ['total_crossed','per_position_crossed','total_persistent_crossing','per_position_persistent_crossing']:
                f=sub.groupby(['group_id','id','regime'])[metric].mean().reset_index().dropna(subset=[metric])
                baseline=f[f.regime=='answer_success'];other=f[f.regime==target]
                row=dict(model=model,relation=relation,regime=target,metric=metric,
                    success_subjects=baseline.group_id.nunique(),regime_subjects=other.group_id.nunique(),difference=None,lo=None,hi=None)
                if len(baseline) and len(other):row['difference']=float(other[metric].mean()-baseline[metric].mean())
                if min(row['success_subjects'],row['regime_subjects'])>=5:
                    ids=f.group_id.unique();draws=[]
                    for _ in range(nboot):
                        counts=pd.Series(rng.choice(ids,len(ids),replace=True)).value_counts();w=f.group_id.map(counts).fillna(0).to_numpy()
                        y=f[metric].to_numpy();flag=f.regime.eq(target).to_numpy()
                        if w[flag].sum() and w[~flag].sum():draws.append(np.average(y[flag],weights=w[flag])-np.average(y[~flag],weights=w[~flag]))
                    if len(draws)>.9*nboot:row['lo'],row['hi']=map(float,np.quantile(draws,[.025,.975]))
                out.append(row)
    return pd.DataFrame(out)


def report(out,cfg):
    out=Path(out);dest=out/'reports';dest.mkdir(exist_ok=True)
    tables={n:[] for n in ['roles','positions','embedding_certificates','jacobian','crossovers','sgd','curvature','coverage']}
    for model in cfg['models']:
        for stage in cfg['stages']:
            for p in sorted((out/model/stage).glob('*.json')):
                r=json.loads(p.read_text());meta={k:r[k] for k in META};data=r['data']
                tables['coverage'].append(dict(**meta,stage=stage,seconds=r['seconds'],margin=r['baseline_margin'],native_certificate=r['native_certificate'],input_origin_error=data.get('input_origin_error') if isinstance(data,dict) else None,input_affine_bias=data.get('input_affine_bias') if isinstance(data,dict) else None))
                if stage=='adjoints':
                    for name,field in [('roles','roles'),('positions','positions'),('embedding_certificates','finite_checks')]:tables[name].extend(dict(**meta,**v) for v in data[field])
                elif stage=='jacobian':
                    tables['jacobian'].extend(dict(**meta,**{k:v for k,v in x.items() if k!='probe_sq'}) for x in data['rows'])
                    tables['crossovers'].append(dict(**meta,**data['crossover']))
                else:tables[stage].extend(dict(**meta,**v) for v in data)
    for name,rows in tables.items():
        if rows:pd.DataFrame(rows).to_csv(dest/(name+'_per_prompt.csv'),index=False)
    if tables['roles']:
        df=pd.DataFrame(tables['roles']);all_df=df.assign(regime='ALL');combined=pd.concat([df,all_df])
        summary=summarize(combined,['model','regime','role','layer'],METRICS,cfg['bootstrap']);summary.to_csv(dest/'adjoint_summary.csv',index=False)
        summarize(combined[combined.layer==0],['model','regime','role'],METRICS+['frozen_origin_margin'],cfg['bootstrap']).to_csv(dest/'input_origin_summary.csv',index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for model in df.model.unique():
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for role in ['subject','template','answer']:
                d=summary[(summary.model==model)&(summary.regime=='ALL')&(summary.role==role)].sort_values('layer')
                for ax,metric in zip(axes,['angle_degrees','selection_fraction']):
                    ax.plot(d.layer,d[metric],label=role);ax.fill_between(d.layer.to_numpy(dtype=float),pd.to_numeric(d[metric+'_lo'],errors='coerce').to_numpy(dtype=float),pd.to_numeric(d[metric+'_hi'],errors='coerce').to_numpy(dtype=float),alpha=.12)
                    ax.set(xlabel='Residual cut (0=input; L=pre-final-norm)',ylabel=metric);ax.legend()
            fig.suptitle(model+' — descriptive adjoint comparison');fig.tight_layout();fig.savefig(dest/(model+'_adjoints.png'),dpi=170);fig.savefig(dest/(model+'_adjoints.pdf'));plt.close(fig)
    if tables['crossovers']:
        df=pd.DataFrame(tables['crossovers'])
        metrics=['total_crossed','per_position_crossed','total_persistent_crossing','per_position_persistent_crossing']
        for m in metrics:df[m]=pd.to_numeric(df[m],errors='coerce')
        summarize(df,['model','regime'],metrics,cfg['bootstrap']).to_csv(dest/'crossover_summary.csv',index=False)
        relation_contrasts(df,cfg['bootstrap']).to_csv(dest/'crossover_within_relation.csv',index=False)
    if tables['sgd']:
        df=pd.DataFrame(tables['sgd']);df['gold_top1']=df.gold_top1.astype(float)
        summarize(df,['model','regime','layer','family','dose'],['live_gain','frozen_gain','content_linear','selection_linear','finite_reselection','live_nonlinear_remainder','gold_top1'],cfg['bootstrap']).to_csv(dest/'sgd_summary.csv',index=False)
    checks=tables['embedding_certificates'];updates=tables['sgd']
    dump(dest/'audit.json',dict(completed_stage_prompts=len(tables['coverage']),embedding_checks=len(checks),
        embedding_failures=sum(not r['passed'] for r in checks),embedding_unresolved=sum(not r['resolved'] for r in checks),
        frozen_update_checks=len(updates),frozen_update_failures=sum(not r['frozen_linear_pass'] for r in updates),
        maximum_native_certificate=max([r['native_certificate'] for r in tables['coverage']],default=0.)))
    dump(dest/'interpretation.json',dict(
        cuts='Block-entry x_l for l=0..L-1, plus pre-final-norm cut L. Final norm may prevent late cosine convergence.',
        roles='subject/first/template partition tokens, with subject priority over first. subject_last and answer are overlapping views, NOT extra origins. Answer means last prompt position, not generated answer embedding.',
        selection_fraction='||live-frozen||/(||live-frozen||+||frozen||). Bounded descriptive norm ratio, not additive energy/causal percentage. Cross term is saved.',
        jacobian='Gate=SiLU(gate_proj(normalized MLP input)); raw and per-position block norms. Probes estimate squared Frobenius norms. First and 3-consecutive crossings saved; missing means no crossing, never last layer.',
        crossover='Within-relation tables compare origin regimes with answer_success. Depth conditioned on crossing; report crossing rate alongside depth. Probe SE is Monte Carlo uncertainty, separate from subject bootstrap. No causal-stage validation.',
        sgd='One matrix per intervention, first canonical gold-token CE, fixed baseline strongest nongold competitor. Norm-relative steps with equivalent SGD learning rate saved. Frozen finite gain checked against exact single-matrix linear prediction; finite reselection is live minus frozen replay. No sequence-level repair or specificity claim.',
        curvature='One matrix at a time, baseline margin-gradient direction. Frozen same-matrix Hessian should vanish. Multi-matrix cross-curvature is not covered.',
        inference='Bootstrap clusters subjects after equal-fact averaging, conditional on fixed cohort/reference labels/models. Pointwise intervals, no multiplicity adjustment. Sparse regimes/uncalibrated cases retained.'))
