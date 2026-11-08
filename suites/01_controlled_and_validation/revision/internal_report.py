"""Cross-label internal repair matrix with paired controls and Holm adjustment."""
import argparse
import numpy as np
import pandas as pd
from revision.common import manifest,bootstrap_mean


def holm(p):
    p=np.asarray(p); order=np.argsort(p); out=np.empty(len(p)); maximum=0.
    for rank,i in enumerate(order):
        maximum=max(maximum,(len(p)-rank)*p[i]); out[i]=min(1.,maximum)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True); p.add_argument('--repairs',required=True)
    p.add_argument('--out',required=True); p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--permutations',type=int,default=5000); p.add_argument('--seed',type=int,default=0)
    args=p.parse_args(); out=manifest(args.out,args,[args.features,args.repairs])
    labels=pd.read_csv(args.features,dtype={'id':str,'group_id':str})[['id','label']]
    repairs=pd.read_csv(args.repairs,dtype={'id':str,'group_id':str})
    if labels.id.duplicated().any(): p.error('Duplicate labels')
    df=repairs.merge(labels,on='id',validate='many_to_one'); rows=[]; comparisons=[]
    if df.empty: p.error('No matched repair outcomes')
    keys=['model','layer','alpha','label']
    for key,sub in df.groupby(keys):
        meta=dict(zip(keys,key))
        for arm,s in sub.groupby('arm'):
            rows.append(dict(**meta,arm=arm,**bootstrap_mean(s.recovered.astype(float),s.group_id,args.bootstrap,args.seed)))
        for control in ['random_norm_matched','wrong_donor_norm_matched']:
            selected=sub[sub.arm.isin(['donor',control])]
            pivot=selected.pivot(index=['id','group_id'],columns='arm',values='recovered')
            if 'donor' not in pivot or control not in pivot: continue
            pivot=pivot.dropna(); diff=pivot.donor.astype(float)-pivot[control].astype(float)
            # Equal-group effect and group sign-flip randomization, two-sided.
            grouped=diff.groupby(level='group_id').mean(); v=grouped.to_numpy()
            if not len(v): continue
            rng=np.random.default_rng(args.seed); observed=abs(v.mean())
            exceeds=sum(abs((v*rng.choice([-1,1],size=len(v))).mean())>=observed-1e-12
                        for _ in range(args.permutations))
            comparisons.append(dict(**meta,control=control,
                p=(exceeds+1)/(args.permutations+1),
                **bootstrap_mean(v,grouped.index,args.bootstrap,args.seed)))
    pd.DataFrame(rows).to_csv(out/'recovery_matrix.csv',index=False)
    comp=pd.DataFrame(comparisons)
    if len(comp): comp['p_holm']=holm(comp.p); comp.to_csv(out/'paired_controls.csv',index=False)
    print('CIs are descriptive; Holm adjusts all exported paired-control tests together.')


if __name__=='__main__': main()
