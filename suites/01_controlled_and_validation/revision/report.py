"""Vector plots and cluster uncertainty; no hard-coded expected findings."""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from revision.common import manifest,bootstrap_mean


def ranking_report(path,out,boots,seed):
    df=pd.read_csv(path); toy='condition' in df
    groupcols=['seed','condition','example'] if toy else ['model','id','group_id']
    methods=[m for m in ['affine_edge_sum','input_provenance','attention','legacy_gradxact','gate_gradient','gate_ig'] if m in df]
    rows=[]
    for key,sub in df.groupby(groupcols):
        meta=dict(zip(groupcols,key)); y=sub.exact_drop.to_numpy()
        for method in methods:
            x=sub[method].to_numpy(); order=np.argsort(-x,kind='stable'); oracle=np.argsort(-y,kind='stable')
            top=order[0]; k=min(8,len(y)); denom=y[oracle[:k]].sum()
            rho=float(spearmanr(x,y).statistic) if len(set(x))>1 and len(set(y))>1 else np.nan
            item=dict(**meta,method=method,rho=rho,
                capture8=float(y[order[:k]].sum()/denom) if denom>1e-9 else np.nan,
                oracle_recall8=len(set(order[:k])&set(oracle[:k]))/k,
                sign_accuracy=float(np.mean(np.sign(x)==np.sign(y))),
                n_candidates=len(y))
            if toy:
                item.update(hit_strict=float(sub.iloc[top].true_source),
                            hit_answer_position=float(sub.iloc[top].answer_token_position),
                            correct=bool(sub.iloc[0].correct))
            rows.append(item)
    per=pd.DataFrame(rows); per.to_csv(out/'per_example.csv',index=False)
    summaries=[]
    for condition,sub in per.groupby('condition' if toy else 'model'):
        for method,s in sub.groupby('method'):
            for metric in ['rho','capture8','oracle_recall8','sign_accuracy']+(['hit_strict','hit_answer_position'] if toy else []):
                # Equal-seed estimand for toy training runs; cluster by fact for natural data.
                if toy:
                    means=s.groupby('seed')[metric].mean(); values=means.values; groups=means.index.astype(str)
                else: values=s[metric]; groups=s.group_id
                estimate=bootstrap_mean(values,groups,boots,seed)
                if len(set(groups))<2: estimate['lo']=None; estimate['hi']=None
                summaries.append(dict(condition=condition,method=method,metric=metric,**estimate))
    summary=pd.DataFrame(summaries); summary.to_csv(out/'ranking_summary.csv',index=False)
    # Paired method differences at the same unit of resampling.
    differences=[]
    for condition,sub in per.groupby('condition' if toy else 'model'):
        for baseline in [m for m in methods if m!='affine_edge_sum']:
            idx=['seed','example'] if toy else ['id','group_id']
            pivot=sub.pivot(index=idx,columns='method',values='rho').dropna(subset=['affine_edge_sum',baseline])
            d=pivot['affine_edge_sum']-pivot[baseline]
            if toy: d=d.groupby(level='seed').mean(); groups=d.index.astype(str)
            else: groups=d.index.get_level_values('group_id')
            estimate=bootstrap_mean(d.values,groups,boots,seed)
            if len(set(groups))<2: estimate['lo']=None; estimate['hi']=None
            differences.append(dict(condition=condition,comparison=f'affine minus {baseline}',**estimate))
    pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
    checks=df[['ig_residual','ig_refinement','gradient_fd_error']].abs().describe()
    checks.to_csv(out/'numerical_checks.csv')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates'); p.add_argument('--robustness'); p.add_argument('--out',required=True)
    p.add_argument('--bootstrap',type=int,default=2000); p.add_argument('--seed',type=int,default=0)
    args=p.parse_args(); paths=[a for a in [args.candidates,args.robustness] if a]
    if not paths: p.error('Provide candidates or robustness summary')
    out=manifest(args.out,args,paths)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':10})
    if args.candidates:
        summary=ranking_report(args.candidates,out,args.bootstrap,args.seed)
        sub=summary[summary.metric=='rho']; conditions=sub.condition.unique()
        fig,axes=plt.subplots(1,len(conditions),figsize=(4*len(conditions),4),squeeze=False)
        for ax,cond in zip(axes[0],conditions):
            s=sub[sub.condition==cond]; y=np.arange(len(s))
            ax.scatter(s['mean'],y)
            for j,(_,r) in enumerate(s.iterrows()):
                if pd.notna(r.lo) and pd.notna(r.hi): ax.plot([r.lo,r.hi],[j,j],color='C0')
            ax.set_yticks(y,s.method); ax.set_xlim(-1,1); ax.axvline(0,color='grey',lw=.5)
            ax.set_title(str(cond)); ax.set_xlabel('Spearman with measured removal effect')
        fig.tight_layout(); fig.savefig(out/'ranking.svg'); fig.savefig(out/'ranking.pdf'); plt.close(fig)
    if args.robustness:
        df=pd.read_csv(args.robustness)
        # One declared slice; all results remain in CSV. Avoid averaging distinct experimental choices.
        sl=df[(df.span=='original')&(df.policy=='random')&(df['mode']=='within_task')&(df.consecutive==2)]
        sl=sl[(sl.decoy_size==sl.decoy_size.min())&(sl.calibration_n==sl.calibration_n.max())]
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,lab in zip(axes,['source','transport','selection']):
            for definition,s in sl[sl.variable==lab].groupby('definition'):
                s=s.sort_values('quantile'); ax.plot(s['quantile'],s['mean'],marker='o',label=definition)
                ax.fill_between(s['quantile'],s.lo,s.hi,alpha=.15)
            ax.set_ylim(0,1); ax.set_title(lab); ax.set_xlabel('Calibration quantile')
        axes[0].set_ylabel('Share among evaluated failures'); axes[-1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out/'robustness.svg'); fig.savefig(out/'robustness.pdf'); plt.close(fig)


if __name__=='__main__': main()
