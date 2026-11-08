"""Trained toy retrieval benchmark with paired conditions and complete candidate evaluation."""
import argparse
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from known_circuit import make_model
from known_circuit2 import build_vocab, sample_seq, decoy_edit, _manual_edited_cache
from frozen_cache import Weights, certify_frozen, tdla_edge_scores
from revision.common import manifest, seed_all, write_json
from revision.measure import provenance, partition_certificate
from revision.baselines import matched_scores,identity_certificate,legacy_gradxact


def train(seed,steps,device,path):
    seed_all(seed); vocab,tid=build_vocab(); model=make_model(len(vocab),device)
    if path.exists():
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    else:
        rng=random.Random(seed); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        model.train()
        for step in range(steps):
            batch=[]
            for _ in range(64):
                seq,ans,_,_=sample_seq(rng,tid,competing=rng.random()<.5); batch.append(seq+[ans])
            x=torch.tensor(batch,device=device); labels=torch.full_like(x,-100); labels[:,-1]=x[:,-1]
            loss=model(x,labels=labels).loss; loss.backward(); opt.step(); opt.zero_grad()
            if step%100==0: print(f'Seed {seed}, training {step}/{steps}: loss {float(loss.detach()):.3f}',flush=True)
        torch.save(model.state_dict(),path)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,tid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--seeds',type=int,default=12); p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--examples',type=int,default=60); p.add_argument('--ig-steps',type=int,default=16)
    p.add_argument('--threads',type=int,default=1); args=p.parse_args()
    out=manifest(args.out,args); torch.set_num_threads(args.threads)
    (out/'checkpoints').mkdir(exist_ok=True); (out/'examples').mkdir(exist_ok=True)
    for seed in range(args.seeds):
        model,tid=train(seed,args.steps,args.device,out/'checkpoints'/f'{seed}.pt'); W=Weights(model)
        for competing in (False,True):
            # Exactly the same prompts in easy/decoy and competing/hard.
            rng=random.Random(100000+seed+1000*competing)
            prompts=[sample_seq(rng,tid,competing) for _ in range(args.examples)]
            for decoy in (False,True):
                condition=('hard' if competing else 'decoy') if decoy else ('competing' if competing else 'easy')
                edit=decoy_edit if decoy else None
                for i,(seq,g,true,alt) in enumerate(prompts):
                    path=out/'examples'/f'{seed}_{condition}_{i}.json'
                    if path.exists(): continue
                    ids=torch.tensor(seq,device=args.device)
                    with torch.no_grad(): C=_manual_edited_cache(model,W,ids,edit)
                    certify_frozen(W,C); partition_certificate(W,C,[true]); identity_certificate(model,W,ids,edit)
                    z=C.logits[-1]; c=int(z.argmax())
                    if c==g: c=int(z.topk(2).indices[1])
                    columns=list(range(len(seq)-1))
                    start=time.perf_counter()
                    affine=tdla_edge_scores(W,C,g,columns,tok_c=c).sum((0,1)).cpu().tolist()
                    affine_seconds=time.perf_counter()-start
                    attn=torch.stack([lc.A[:,-1,:-1].sum(0) for lc in C.layers]).sum(0).cpu().tolist()
                    legacy=legacy_gradxact(model,W,C,ids,g,c,[[col] for col in columns],edit).sum((1,2)).cpu().tolist()
                    rows=[]
                    for col in columns:
                        scores=matched_scores(model,W,ids,g,c,[col],steps=args.ig_steps,base_edit=edit)
                        input_margin=provenance(W,C,[col],g)-provenance(W,C,[col],c)
                        rows.append(dict(seed=seed,condition=condition,example=i,column=col,
                            true_source=col==true,answer_token_position=col in alt,correct=int(z.argmax())==g,
                            affine_edge_sum=affine[col],input_provenance=input_margin,attention=attn[col],
                            legacy_gradxact=legacy[col],
                            affine_total_seconds=affine_seconds,**scores))
                    write_json(path,rows)
                    print(f'{seed}/{condition}/{i} saved',flush=True)
    import json
    rows=[r for f in sorted((out/'examples').glob('*.json')) for r in json.loads(f.read_text())]
    pd.DataFrame(rows).to_csv(out/'candidates.csv',index=False)
    print('All examples retained. Source positions are task annotations, not verified causal circuits.')


if __name__=='__main__': main()
