"""Matched edge-removal ranking on natural cohorts. Explicit random candidate subsampling."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from revision.common import manifest,read_jsonl,validate_records,digest,write_json,seed_all
from revision.measure import supported
from revision.baselines import matched_scores,identity_certificate,legacy_gradxact
from frozen_cache import tdla_edge_scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True); p.add_argument('--cohort',required=True)
    p.add_argument('--out',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--candidates',type=int,default=0,help='0=all; positive=uniform random subset, chosen without scores')
    p.add_argument('--ig-steps',type=int,default=16); p.add_argument('--seed',type=int,default=0)
    args=p.parse_args(); records=read_jsonl(args.cohort); validate_records(records)
    out=manifest(args.out,args,[args.cohort]); (out/'examples').mkdir(exist_ok=True); seed_all(args.seed)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,
           attn_implementation='eager').to(args.device)
    W,api=supported(model)
    if model.config.model_type=='gpt_neox': p.error('Live attention gate adapter supports Llama/Qwen2 only')
    for i,r in enumerate(records):
        if r['split']!='evaluation': continue
        path=out/'examples'/f'{digest(str(r["id"]))}.json'
        if path.exists(): continue
        ids=torch.tensor(r['input_ids'],device=args.device) if 'input_ids' in r else tok(r['prompt'],return_tensors='pt').input_ids[0].to(args.device)
        C=api.build(ids); api.certify(C); identity_certificate(model,W,ids)
        S=list(range(r['source_span'][0],r['source_span'][1]+1))
        if len(S)>=len(ids): raise ValueError('Cannot renormalize removing all source columns')
        g=r['target_first_token']; z=C.logits[-1]; c=int(z.argmax())
        if c==g: c=int(z.topk(2).indices[1])
        scores=tdla_edge_scores(W,C,g,S,tok_c=c).sum(-1)
        legacy=legacy_gradxact(model,W,C,ids,g,c,[S])[0]
        candidates=[(l,h) for l in range(W.L) for h in range(W.H)]
        if args.candidates:
            rng=np.random.default_rng(args.seed+i)
            candidates=[candidates[j] for j in sorted(rng.choice(len(candidates),min(args.candidates,len(candidates)),replace=False))]
        rows=[]
        for l,h in candidates:
            sc=matched_scores(model,W,ids,g,c,S,(l,h),args.ig_steps)
            rows.append(dict(id=str(r['id']),group_id=str(r['group_id']),model=args.model,layer=l,head=h,
                candidate_universe='all' if not args.candidates else 'uniform_subset',
                total_candidates=W.L*W.H,measured_candidates=len(candidates),
                affine_edge_sum=float(scores[l,h]),legacy_gradxact=float(legacy[l,h]),attention=float(C.layers[l].A[h,-1,S].sum()),**sc))
        write_json(path,rows); print(f'Saved {r["id"]}',flush=True)
    rows=[r for f in (out/'examples').glob('*.json') for r in json.loads(f.read_text())]
    pd.DataFrame(rows).to_csv(out/'candidates.csv',index=False)


if __name__=='__main__': main()
