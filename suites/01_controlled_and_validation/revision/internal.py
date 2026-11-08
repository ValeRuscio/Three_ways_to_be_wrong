"""Verified-donor residual patching with AtP and candidate-specific path IG.

Sites are residual inputs at each layer over the declared source span. Donor
and receiver spans must be explicitly aligned. Wrong-answer and random-vector
controls receive the SAME residual displacement norm as the true donor.
This tests repair associations, not a guaranteed exclusive stage taxonomy.
"""
import argparse
from contextlib import contextmanager
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from revision.common import manifest,read_jsonl,validate_records,digest,write_json,seed_all
from revision.measure import supported


@contextmanager
def patch(model,layer,S,delta,alpha):
    blocks=model.gpt_neox.layers if model.config.model_type=='gpt_neox' else model.model.layers
    def hook(module,args,kw):
        x=args[0] if args else kw['hidden_states']
        new=x.clone(); new[:,S]=x[:,S]+alpha*delta[None]
        if args: return (new,*args[1:]),kw
        return args,dict(kw,hidden_states=new)
    handle=blocks[layer].register_forward_pre_hook(hook,with_kwargs=True)
    try: yield
    finally: handle.remove()


def patch_measure(model,ids,g,c,l,S,delta,alpha,grad=False):
    with torch.set_grad_enabled(grad):
        a=torch.tensor(float(alpha),device=ids.device,requires_grad=grad)
        with patch(model,l,S,delta,a): z=model(ids[None],use_cache=False).logits[0,-1]
        m=z[g]-z[c]
        if grad: return float(torch.autograd.grad(m,a)[0])
        return dict(margin=float(m),recovered=int(z.argmax())==g)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True); p.add_argument('--cohort',required=True)
    p.add_argument('--out',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--layers',type=int,nargs='+'); p.add_argument('--alphas',type=float,nargs='+',default=[.25,.5,1.])
    p.add_argument('--ig-steps',type=int,default=16); p.add_argument('--seed',type=int,default=0)
    args=p.parse_args(); records=read_jsonl(args.cohort); validate_records(records)
    if args.ig_steps<2: p.error('At least 2 IG intervals')
    out=manifest(args.out,args,[args.cohort]); (out/'examples').mkdir(exist_ok=True); seed_all(args.seed)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,
        attn_implementation='eager').to(args.device); W,api=supported(model)
    layers=args.layers if args.layers is not None else list(range(W.L))
    if any(l<0 or l>=W.L for l in layers): p.error('Invalid layer')
    def ids_for(r,prefix=''):
        k=prefix+'input_ids'
        return torch.tensor(r[k],device=args.device) if k in r else tok(r[prefix+'prompt'],return_tensors='pt').input_ids[0].to(args.device)
    for i,r in enumerate(records):
        if r['split']!='evaluation' or r['correct']: continue
        path=out/'examples'/f'{digest(str(r["id"]))}.json'
        if path.exists(): continue
        ids=ids_for(r); C=api.build(ids); api.certify(C)
        donor_ids=ids_for(r,'donor_'); D=api.build(donor_ids); api.certify(D)
        g=r['target_first_token']; c=int(C.logits[-1].argmax())
        if c==g: raise ValueError('Receiver not failed at declared decision point')
        if int(D.logits[-1].argmax())!=g: raise ValueError(f'Donor is not successful: {r["id"]}')
        S=list(range(r['source_span'][0],r['source_span'][1]+1))
        DS=list(range(r['donor_source_span'][0],r['donor_source_span'][1]+1))
        if len(S)!=len(DS): raise ValueError('Donor span length mismatch; explicit alignment required')
        wrong=None; WS=None
        if 'wrong_donor_prompt' in r or 'wrong_donor_input_ids' in r:
            wrong=api.build(ids_for(r,'wrong_donor_')); api.certify(wrong)
            WS=list(range(r['wrong_donor_source_span'][0],r['wrong_donor_source_span'][1]+1))
            if len(WS)!=len(S): raise ValueError('Wrong donor span length mismatch')
            if int(wrong.logits[-1].argmax())==g: raise ValueError('Wrong donor predicts gold')
        rows=[]; base=float(C.logits[-1,g]-C.logits[-1,c])
        for l in layers:
            delta=(D.resid[l][DS]-C.resid[l][S]).detach()
            norm=float(delta.norm())
            gen=torch.Generator().manual_seed(args.seed+i*W.L+l)
            rand=torch.randn(delta.shape,generator=gen).to(delta); rand*=norm/rand.norm().clamp_min(1e-12)
            arms={'donor':delta,'random_norm_matched':rand}
            if wrong is not None:
                wd=(wrong.resid[l][WS]-C.resid[l][S]).detach()
                arms['wrong_donor_norm_matched']=wd*norm/wd.norm().clamp_min(1e-12)
            for name,d in arms.items():
                identity=patch_measure(model,ids,g,c,l,S,d,0.)['margin']
                if abs(identity-base)>1e-4*max(1,abs(base)): raise AssertionError('Residual patch identity failed')
                atp=patch_measure(model,ids,g,c,l,S,d,0.,True)
                grad=[patch_measure(model,ids,g,c,l,S,d,float(a),True) for a in np.linspace(0,1,2*args.ig_steps+1)]
                ig=float(np.trapezoid(grad,dx=1/(2*args.ig_steps)))
                coarse=float(np.trapezoid(grad[::2],dx=1/args.ig_steps))
                exact=patch_measure(model,ids,g,c,l,S,d,1.)['margin']-base
                for alpha in args.alphas:
                    result=patch_measure(model,ids,g,c,l,S,d,alpha)
                    rows.append(dict(id=str(r['id']),group_id=str(r['group_id']),model=args.model,
                        layer=l,arm=name,alpha=alpha,dose_norm=abs(alpha)*float(d.norm()),
                        margin_gain=result['margin']-base,recovered=result['recovered'],
                        atp_full_patch_gain=atp,ig_full_patch_gain=ig,exact_full_patch_gain=exact,
                        ig_residual=ig-exact,ig_refinement=abs(ig-coarse)))
        write_json(path,rows); print(f'Saved donor repairs: {r["id"]}',flush=True)
    rows=[r for f in (out/'examples').glob('*.json') for r in json.loads(f.read_text())]
    pd.DataFrame(rows).to_csv(out/'internal_repairs.csv',index=False)


if __name__=='__main__': main()
