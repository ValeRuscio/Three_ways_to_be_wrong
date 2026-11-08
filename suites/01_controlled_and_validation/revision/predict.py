"""Grouped held-out utility, paired uncertainty, and label-specific repair response."""
import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from revision.common import manifest, bootstrap_mean, write_json


def grouped_folds(groups, k=5, seed=0):
    groups=np.asarray(groups).astype(str); unique=np.unique(groups)
    if len(unique)<k: raise ValueError(f'Need at least {k} independent groups')
    rng=np.random.default_rng(seed); rng.shuffle(unique)
    for i in range(k):
        test=np.isin(groups,unique[i::k]); yield np.flatnonzero(~test),np.flatnonzero(test)


def cv_predict(frame, numeric, categorical, y, groups, k, seed, regression=False):
    pred=np.zeros(len(frame)); fold=np.zeros(len(frame),dtype=int); notes=[]
    for fi,(tr,te) in enumerate(grouped_folds(groups,k,seed)):
        if not regression and len(np.unique(y[tr]))<2:
            pred[te]=(y[tr].sum()+.5)/(len(tr)+1)
            notes.append(f'Fold {fi}: one-class training; smoothed prevalence fallback')
            fold[te]=fi; continue
        numeric_pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())
        transform=ColumnTransformer([('num',numeric_pipe,numeric),
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
        estimator=Ridge(alpha=1.) if regression else LogisticRegression(C=1.,max_iter=3000)
        pipe=make_pipeline(transform,estimator)
        pipe.fit(frame.iloc[tr],y[tr])
        pred[te]=pipe.predict(frame.iloc[te]) if regression else pipe.predict_proba(frame.iloc[te])[:,1]
        fold[te]=fi
    return pred,fold,notes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',nargs='+',required=True); p.add_argument('--outcomes',nargs='+',required=True)
    p.add_argument('--out',required=True); p.add_argument('--arm',default='answer_up')
    p.add_argument('--target',default='recovered'); p.add_argument('--regression',action='store_true')
    p.add_argument('--folds',type=int,default=5); p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args(); out=manifest(args.out,args,args.features+args.outcomes)
    if len(args.features)!=len(args.outcomes): p.error('Match each features file to one outcomes file')
    frames=[]
    for fi,oi in zip(args.features,args.outcomes):
        f=pd.read_csv(fi,dtype={'id':str,'group_id':str}); o=pd.read_csv(oi,dtype={'id':str})
        if 'arm' in o: o=o[o.arm==args.arm]
        if f.id.duplicated().any() or o.id.duplicated().any(): p.error('Duplicate IDs in join')
        data=f.merge(o,on='id',validate='one_to_one',suffixes=('','_outcome'))
        frames.append(data)
    df=pd.concat(frames,ignore_index=True)
    if args.target not in df: p.error('Missing requested outcome')
    df=df.dropna(subset=[args.target]).reset_index(drop=True)
    if df.empty: p.error('No matched observed outcomes')
    y=df[args.target].astype(float).to_numpy()
    if not args.regression and not set(y)<= {0.,1.}: p.error('Binary target required unless --regression')
    basic=['target_logit','competitor_logit','margin','target_rank','entropy','target_logprob']
    dose=[c for c in ['dose_norm','delta_gold','delta_margin'] if c in df]
    tiers={'logits':basic,'logits_dose':basic+dose,
           'logits_dose_source':basic+dose+['q','attention_mass'],
           'logits_dose_source_transport':basic+dose+['q','attention_mass','tau','share']}
    categories=[c for c in ['model','task','relation'] if c in df]
    for c in categories: df[c]=df[c].fillna('unknown').astype(str)
    # Same fact/group across models remains together. User must supply global group IDs.
    groups=df.group_id.astype(str).to_numpy(); predictions={}; results=[]; all_notes=[]
    for name,cols in tiers.items():
        pred,fold,notes=cv_predict(df,cols,categories,y,groups,args.folds,args.seed,args.regression)
        predictions[name]=pred; all_notes.extend(notes)
        item=dict(tier=name,n=len(y),n_groups=len(set(groups)),mse=mean_squared_error(y,pred))
        if not args.regression:
            item.update(auc=float(roc_auc_score(y,pred)) if len(set(y))==2 else None,
                        log_loss=float(log_loss(y,np.clip(pred,1e-8,1-1e-8),labels=[0,1])))
        results.append(item)
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    predframe=df[['id','group_id','model','label']].copy(); predframe['outcome']=y; predframe['fold']=fold
    for name,v in predictions.items(): predframe[name]=v
    predframe.to_csv(out/'heldout_predictions.csv',index=False)
    comparisons=[]
    for a,b in [('logits','logits_dose_source_transport'),('logits_dose','logits_dose_source_transport'),
                ('logits_dose_source','logits_dose_source_transport')]:
        gain=(y-predictions[a])**2-(y-predictions[b])**2
        comparisons.append(dict(baseline=a,augmented=b,metric='MSE reduction (positive favors augmented)',
            **bootstrap_mean(gain,groups,args.bootstrap,args.seed)))
    write_json(out/'paired_improvement.json',comparisons)
    if not args.regression:
        responses=[]
        for lab,sub in df.groupby('label'):
            responses.append(dict(label=lab,**bootstrap_mean(sub[args.target].astype(float),sub.group_id,args.bootstrap,args.seed)))
        pd.DataFrame(responses).to_csv(out/'repair_by_label.csv',index=False)
    write_json(out/'notes.json',dict(folds=all_notes,
        uncertainty='Cluster bootstrap of fixed out-of-fold losses; excludes model-refitting uncertainty',
        scope='Prediction is not identification of causal error stages. Final-readout outcomes are functions of logit geometry.',
        missing_outcome_rows='Excluded, never imputed. Inspect joined sample counts.'))
    print(pd.DataFrame(results).to_string(index=False))


if __name__=='__main__': main()
