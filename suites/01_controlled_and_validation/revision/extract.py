"""Measure one example at a time; resume atomic per-record files. No activation hoarding."""
import argparse
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from revision.common import read_jsonl, validate_records, manifest, digest, write_json, seed_all
from revision.measure import supported, provenance, partition_certificate, lens_matrix, readout_repairs


def spans(r, T, seed):
    a,b = r['source_span']; base = list(range(a,b+1))
    rng = np.random.default_rng(seed)
    starts = [j for j in range(T-len(base)+1) if not set(range(j,j+len(base))) & set(base)]
    out = dict(original=base, wider=list(range(max(0,a-1),min(T,b+2))),
               first=[a], last=[b])
    if starts:
        j = int(rng.choice(starts)); out['random_control'] = list(range(j,j+len(base)))
    return out


def measure_record(model, tok, W, api, r, pools, args, seed):
    start = time.perf_counter()
    if 'input_ids' in r:
        ids = torch.tensor(r['input_ids'], device=args.device)
    else:
        ids = tok(r['prompt'], return_tensors='pt').input_ids[0].to(args.device)
    g = int(r['target_first_token']); T=len(ids)
    if r['source_span'][1] >= T: raise ValueError('Source span exceeds tokenized prompt')
    C=api.build(ids); error=api.certify(C)
    z=C.logits[-1]; top=int(z.argmax())
    if r['correct'] and top != g:
        raise ValueError('Successful calibration/evaluation record is not top-1 correct at this decision point')
    if not r['correct'] and top == g:
        raise ValueError('Failure is not at this decision point; provide aligned divergence input_ids and target')
    c = int(r.get('competitor_token', top))
    if r['correct']: c=int(z.topk(2).indices[1])
    elif c != top: raise ValueError('Competitor must match actual top token at this decision point')
    key=str(r['relation'])
    if key not in pools: raise ValueError(f'No independent decoy pool for relation {key}')
    decoys=sorted(set(int(d) for d in pools[key])-{g}-set(r.get('excluded_decoy_tokens', [])))
    if not decoys: raise ValueError('Empty decoy set would vacuously pass source readability')
    if min(decoys)<0 or max(decoys)>=len(z): raise ValueError('Decoy token outside vocabulary')
    tokens=[g]+decoys
    xp=api.final_state(C)
    lp=z.log_softmax(-1)
    record={k:r[k] for k in ['id','group_id','split','correct','relation','task','decision_scope']}
    record.update(model=args.model, target=g, competitor=c, input_ids=ids.tolist(),
        decoy_tokens=decoys, target_logit=float(z[g]), competitor_logit=float(z[c]),
        margin=float(z[g]-z[c]), target_rank=int((z>z[g]).sum())+1,
        entropy=float(-(lp.exp()*lp).sum()), target_logprob=float(lp[g]),
        proj_g=float(xp@api.WU[g]/api.WU[g].norm()),
        proj_c=float(xp@api.WU[c]/api.WU[c].norm()), certificate_error=error,
        variants={}, decision_matches_metadata=True)
    with torch.no_grad():
        for name,S in spans(r,T,seed).items():
            cert=partition_certificate(W,C,S)
            vals=lens_matrix(api,C,S,tokens)
            # This is intentionally kept as a separate legacy definition.
            legacy=float(api.tdla(C,g,S).sum())
            cuts=sorted(set([0]+[int(f*W.L) for f in args.cut_fractions]))
            components={str(cut):provenance(W,C,S,g,cut=cut) for cut in cuts}
            identity=bool(r['task']=='extraction' and g in ids[S].tolist())
            record['variants'][name]=dict(positions=S, token_identity=identity,
                lens=vals.tolist(), legacy_edge_sum=legacy, provenance=components,
                partition_error=cert,
                attention_mass=float(sum(lc.A[:,-1,S].sum() for lc in C.layers)))
    record['seconds']=time.perf_counter()-start
    return record, C


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True); p.add_argument('--cohort',required=True)
    p.add_argument('--decoy-pool',required=True); p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--cut-fractions',type=float,nargs='+',default=[.25,.5,.75])
    p.add_argument('--doses',type=float,nargs='+',default=[.25,.5,1.,2.])
    args=p.parse_args()
    if any(not 0<=f<=1 for f in args.cut_fractions) or any(d<0 for d in args.doses): p.error('Invalid cut or dose')
    records=read_jsonl(args.cohort); validate_records(records)
    if any(r.get('decision_scope') not in ('first_token','aligned_divergence') for r in records):
        p.error('Each record must explicitly declare decision_scope')
    # A run is one model and task: no cross-task threshold mixing.
    if len({r['task'] for r in records}) != 1: p.error('Run tasks separately')
    pools=json.load(open(args.decoy_pool))
    out=manifest(args.out,args,[args.cohort,args.decoy_pool]); seed_all(args.seed)
    (out/'records').mkdir(exist_ok=True); (out/'repairs').mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,
                      attn_implementation='eager').to(args.device).eval()
    W,api=supported(model)
    measured=[]
    for i,r in enumerate(records):
        path=out/'records'/f'{digest(str(r["id"]))}.json'
        if path.exists(): row=json.loads(path.read_text())
        else:
            row,C=measure_record(model,tok,W,api,r,pools,args,args.seed+i)
            write_json(path,row); del C
        measured.append(row)
        print(f'Measured {i+1}/{len(records)}: {r["id"]}',flush=True)
    cal=[r for r in measured if r['split']=='calibration']
    med_g=float(np.median([r['proj_g'] for r in cal])); med_c=float(np.median([r['proj_c'] for r in cal]))
    for r in measured:
        if r['split']!='evaluation' or r['correct']: continue
        path=out/'repairs'/f'{digest(str(r["id"]))}.json'
        if path.exists(): continue
        C=api.build(torch.tensor(r['input_ids'],device=args.device)); api.certify(C)
        arms=readout_repairs(api,C,r['target'],r['competitor'],med_g,med_c,args.doses,args.seed)
        write_json(path,dict(id=r['id'],arms=arms)); del C
    write_json(out/'calibration.json',dict(n=len(cal),median_gold_projection=med_g,median_competitor_projection=med_c))


if __name__=='__main__': main()
