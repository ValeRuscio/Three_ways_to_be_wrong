"""Headline-label robustness, calibration transfer, and paired definition comparisons."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from revision.common import manifest, label, run_score, digest, write_json


def features(r, span, definition, decoy_size, policy, seed, consecutive, eps):
    v=r['variants'][span]
    if '_lens_array' not in v: v['_lens_array']=np.asarray(v['lens'])
    scores=v['_lens_array']; n=scores.shape[-1]-1
    if decoy_size>n: return None # never silently shrink a requested pool
    if policy=='hard':
        # Declared adversarial source test: choose strongest lens decoys across source positions/layers.
        idx=np.argsort(scores[:,:,1:].max(axis=(0,1)))[-decoy_size:]+1
    else:
        rng=np.random.default_rng(int(digest([r['id'],seed])[:12],16))
        idx=rng.choice(n,size=decoy_size,replace=False)+1
    q=run_score(scores[:,:,0]-scores[:,:,idx].max(axis=-1),consecutive)
    tau=v['legacy_edge_sum'] if definition=='legacy_edge_sum' else v['provenance'][definition.split(':')[1]]
    denom=abs(r['target_logit'])
    if denom<=eps: return None # abstain, rather than create enormous transport shares
    return dict(q=q,readable=v['token_identity'] or q>0,tau=tau,share=tau/denom,
                attention_mass=v['attention_mass'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements',required=True); p.add_argument('--out',required=True)
    p.add_argument('--quantiles',type=float,nargs='+',default=[.05,.1,.2])
    p.add_argument('--sizes',type=int,nargs='+',default=[20,40,60])
    p.add_argument('--decoy-sizes',type=int,nargs='+',default=[5,10,20])
    p.add_argument('--repeats',type=int,default=100); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--spans',nargs='+',default=['original','wider','first','last','random_control'])
    p.add_argument('--definitions',nargs='+',default=['provenance:0','legacy_edge_sum'])
    p.add_argument('--policies',nargs='+',default=['random','hard'])
    p.add_argument('--consecutive',type=int,nargs='+',default=[2])
    p.add_argument('--epsilon',type=float,default=1e-6)
    p.add_argument('--transfer',action='store_true',help='Add calibration excluding evaluation relation')
    args=p.parse_args()
    if args.repeats<1 or any(not 0<q<1 for q in args.quantiles): p.error('Invalid repeats or quantile')
    if any(s<1 for s in args.sizes+args.decoy_sizes+args.consecutive): p.error('Sizes must be positive')
    paths=sorted((Path(args.measurements)/'records').glob('*.json'))
    if not paths: p.error('No measurements')
    out=manifest(args.out,args,paths); rows=[json.loads(f.read_text()) for f in paths]
    failures=[r for r in rows if r['split']=='evaluation' and not r['correct']]
    successes=[r for r in rows if r['split']=='calibration']
    if not failures or not successes: p.error('Need failures and calibration successes')
    summaries=[]; stability=[]; transitions=[]; exclusions=[]; reference=[]
    base_config=('original','provenance:0',min(args.decoy_sizes),'random',2)
    def config_features(config,seed):
        span,definition,size,policy,consecutive=config
        data={}
        for r in rows:
            if span not in r['variants']: continue
            f=features(r,span,definition,size,policy,seed,consecutive,args.epsilon)
            if f is not None: data[str(r['id'])]=f
        return data
    base=config_features(base_config,args.seed)
    basecal=[base[str(r['id'])] for r in successes if str(r['id']) in base]
    if not basecal: p.error('Reference configuration has no eligible calibration examples')
    bt,bs=np.quantile([f['tau'] for f in basecal],.1),np.quantile([f['share'] for f in basecal],.1)
    reference_labels={}
    repairs={}
    for f in (Path(args.measurements)/'repairs').glob('*.json'):
        a=json.loads(f.read_text()); repairs[str(a['id'])]=a['arms']
    for r in failures:
        rid=str(r['id'])
        if rid not in base: continue
        f=base[rid]; lab=label(f['readable'],f['tau'],f['share'],bt,bs); reference_labels[rid]=lab
        simple={k:v for k,v in r.items() if k not in ('variants','input_ids','decoy_tokens')}
        simple.update(f,label=lab)
        reference.append(simple)
    pd.DataFrame(reference).to_csv(out/'reference_features.csv',index=False)
    repair_rows=[dict(id=rid,**a) for rid,arms in repairs.items() for a in arms]
    pd.DataFrame(repair_rows).to_csv(out/'repairs.csv',index=False)
    configs=itertools.product(args.spans,args.definitions,args.decoy_sizes,args.policies,args.consecutive)
    labels=['source','transport','selection']
    streamed=['replicates','summary','label_stability','transitions','eligibility']
    for name in streamed: (out/(name+'.csv')).unlink(missing_ok=True)
    def append_frame(frame,name):
        path=out/(name+'.csv')
        frame.to_csv(path,index=False,mode='a',header=not path.exists())
    completed=0
    for ci,config in enumerate(configs):
        summaries=[]; stability=[]; transitions=[]; exclusions=[]
        span,definition,ds,policy,consecutive=config
        modes=['within_task']+(['leave_relation_out'] if args.transfer else [])
        counts={}
        for repeat in range(args.repeats):
            if policy!='hard' or repeat==0:
                data=config_features(config,args.seed+repeat)
            for mode in modes:
                domains=sorted({str(r['relation']) for r in failures}) if mode=='leave_relation_out' else ['all']
                for domain in domains:
                    cal_records=[r for r in successes if str(r['id']) in data
                         and (mode!='leave_relation_out' or str(r['relation'])!=domain)]
                    ev=[r for r in failures if str(r['id']) in data and
                        (domain=='all' or str(r['relation'])==domain)]
                    if not cal_records or not ev: continue
                    blocks={}
                    for r in cal_records:
                        blocks.setdefault(str(r['group_id']),[]).append(data[str(r['id'])])
                    cal=list(blocks.values())
                    for n in sorted(set([len(cal)]+args.sizes)): # full calibration plus requested smaller sizes
                        if n>len(cal): continue
                        rng=np.random.default_rng(args.seed+repeat)
                        sampled=[f for j in rng.integers(len(cal),size=n) for f in cal[j]]
                        for quantile in args.quantiles:
                            tt=float(np.quantile([f['tau'] for f in sampled],quantile))
                            ts=float(np.quantile([f['share'] for f in sampled],quantile))
                            predictions=[]; agree=[]
                            key=(span,definition,ds,policy,consecutive,mode,domain,n,quantile)
                            for r in ev:
                                rid=str(r['id']); f=data[rid]
                                lab=label(f['readable'],f['tau'],f['share'],tt,ts); predictions.append(lab)
                                counts.setdefault((key,rid),dict.fromkeys(labels,0))[lab]+=1
                                if rid in reference_labels: agree.append(lab==reference_labels[rid])
                            summaries.append(dict(span=span,definition=definition,decoy_size=ds,policy=policy,
                                consecutive=consecutive,mode=mode,domain=domain,calibration_n=n,
                                quantile=quantile,repeat=repeat,calibration_examples=len(sampled),n=len(ev),coverage=len(ev)/len(failures),
                                tau_threshold=tt,share_threshold=ts,agreement=np.mean(agree) if agree else np.nan,
                                **{l:predictions.count(l)/len(predictions) for l in labels}))
            if repeat==0:
                exclusions.append(dict(config=str(config),eligible=len(data),total=len(rows)))
        for (key,rid),cnt in counts.items():
            n=sum(cnt.values()); base_lab=reference_labels.get(rid)
            stability.append(dict(config=str(key),id=rid,n=n,modal_stability=max(cnt.values())/n,
                                  reference_agreement=cnt.get(base_lab,0)/n,
                                  **{l:cnt[l]/n for l in labels}))
            for l in labels: transitions.append(dict(config=str(key),reference=base_lab,label=l,count=cnt[l]))
        append_frame(pd.DataFrame(exclusions),'eligibility')
        if summaries:
            completed+=1
            df=pd.DataFrame(summaries)
            append_frame(df,'replicates')
            keys=['span','definition','decoy_size','policy','consecutive','mode','domain','calibration_n','quantile']
            long=df.melt(id_vars=keys+['repeat'],value_vars=labels+['agreement','coverage'])
            aggregate=long.groupby(keys+['variable']).value.agg(mean='mean',lo=lambda x:x.quantile(.025),hi=lambda x:x.quantile(.975)).reset_index()
            append_frame(aggregate,'summary')
            append_frame(pd.DataFrame(stability),'label_stability')
            tr=pd.DataFrame(transitions).groupby(['config','reference','label'],dropna=False)['count'].sum().reset_index()
            append_frame(tr,'transitions')
        print(f'Completed robustness configuration {ci+1}: {config}',flush=True)
    if not completed: p.error('No eligible configurations; check pool/calibration sizes')
    write_json(out/'interpretation.json',dict(reference=dict(span='original',definition='provenance:0',decoy_size=base_config[2],
        policy='random',consecutive=2,quantile=.1),
        interval='Calibration-group-bootstrap and decoy-resampling variability conditional on fixed evaluation cohort; not population CI',
        calibration_n_unit='Independent groups, not rows; calibration_examples records actual sample size',
        transfer='Leave-relation-out uses other relations in the same task; inspect coverage and cohort sizes'))


if __name__=='__main__': main()
