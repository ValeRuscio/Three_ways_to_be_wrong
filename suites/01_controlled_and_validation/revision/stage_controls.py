"""Known-operation controls on trained retrieval models, independent of diagnostic labels.

Operations: corrupt the answer token; suppress its outgoing final-query attention;
add competitor-only output bias. These are intervention labels, NOT exhaustive
causal ground truth: redundancy, failed interventions, and mixed diagnoses remain.
"""
import argparse
import random
import numpy as np
import torch
import pandas as pd
from known_circuit2 import sample_seq,_manual_edited_cache
from revision.controlled import train
from revision.common import manifest,run_score,label,write_json
from revision.measure import supported,provenance,lens_matrix
from revision.baselines import gate_edit


def stats(W,api,C,g,S,decoys):
    scores=lens_matrix(api,C,S,[g]+decoys)
    q=run_score(scores[:,:,0]-scores[:,:,1:].max(-1),2)
    tau=provenance(W,C,S,g); denominator=abs(float(C.logits[-1,g]))
    return dict(q=q,readable=q>0,tau=tau,share=tau/(denominator+1e-9))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--seeds',type=int,default=12); p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--examples',type=int,default=100); p.add_argument('--calibration',type=int,default=100)
    p.add_argument('--max-attempts',type=int,default=10000); args=p.parse_args()
    out=manifest(args.out,args); (out/'checkpoints').mkdir(exist_ok=True); torch.set_num_threads(1)
    rows=[]; counts=[]
    for seed in range(args.seeds):
        result_path=out/f'seed_{seed}.json'
        if result_path.exists():
            import json
            saved=json.loads(result_path.read_text()); rows.extend(saved['rows']); counts.append(saved['counts']); continue
        model,tid=train(seed,args.steps,args.device,out/'checkpoints'/f'{seed}.pt'); W,api=supported(model)
        vals=[v for k,v in tid.items() if k.startswith('V')]; rng=random.Random(900000+seed)
        cal=[]; evaluation=[]; seen=set(); attempts=0
        while len(cal)<args.calibration or len(evaluation)<args.examples:
            if attempts>=args.max_attempts: break
            attempts+=1
            seq,g,true,alt=sample_seq(rng,tid,False)
            if len(alt)!=1 or tuple(seq) in seen: continue
            seen.add(tuple(seq)); ids=torch.tensor(seq,device=args.device)
            C=api.build(ids); api.certify(C)
            if int(C.logits[-1].argmax())!=g: continue
            decoys=[v for v in vals if v!=g]
            if len(cal)<args.calibration: cal.append(stats(W,api,C,g,[true],decoys))
            else: evaluation.append((seq,g,true,decoys))
        info=dict(seed=seed,attempts=attempts,calibration_successes=len(cal),evaluation_successes=len(evaluation))
        counts.append(info); seed_rows=[]
        if len(cal)<args.calibration:
            write_json(result_path,dict(rows=[],counts=info,status='Insufficient successful calibration: no scientific estimate'))
            continue
        tt=float(np.quantile([v['tau'] for v in cal],.1)); ts=float(np.quantile([v['share'] for v in cal],.1))
        for i,(seq,g,true,decoys) in enumerate(evaluation):
            ids=torch.tensor(seq,device=args.device); clean=api.build(ids)
            c=int(clean.logits[-1].topk(2).indices[1])
            corrupted=ids.clone(); corrupted[true]=rng.choice(decoys)
            cases=[('source_token_corruption',corrupted,None),
                   ('transport_gate_suppression',ids,gate_edit([true],None,0.)),
                   ('competitor_output_bias',ids,None)]
            for operation,x,edit in cases:
                with torch.no_grad(): C=_manual_edited_cache(model,W,x,edit) if edit else api.build(x)
                api.certify(C)
                z=C.logits[-1].clone()
                if operation=='competitor_output_bias':
                    # External constant output bias; g logit and transported components unchanged.
                    z[c]=z.max()+1.
                failed=int(z.argmax())!=g
                f=stats(W,api,C,g,[true],decoys)
                verdict=label(f['readable'],f['tau'],f['share'],tt,ts) if failed else 'correct'
                seed_rows.append(dict(seed=seed,example=i,operation=operation,failed=failed,
                    label=verdict,tau_threshold=tt,share_threshold=ts,margin=float(z[g]-z.max()),**f))
        rows.extend(seed_rows); write_json(result_path,dict(rows=seed_rows,counts=info,status='completed'))
    pd.DataFrame(counts).to_csv(out/'coverage.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'operations.csv',index=False)
    if rows:
        df=pd.DataFrame(rows)
        df.groupby(['seed','operation','label']).size().rename('count').reset_index().to_csv(out/'operation_label_counts.csv',index=False)
    print('Report all operations including those that do not cause failure. Inspect coverage.csv.')


if __name__=='__main__': main()
