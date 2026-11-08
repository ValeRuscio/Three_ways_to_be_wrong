"""Endpoint checkpoint displacement, not accumulated training gradients.
Subspace is the tensor-product span {U B V.T}; signed full-displacement
alignment is stored separately. Facts selected before observing outcomes.
"""
import gc,re
from pathlib import Path
import torch
from .core import modules,stats,subtract,projected_energy,delta_dot,live
from .experiments import split
from .io import pin,model_load,load,dump,digest,key,tokenize,generate


def revisions(spec,cfg,path):
 if Path(path).exists():return load(path)
 from huggingface_hub import HfApi
 api=HfApi();refs=api.list_repo_refs(spec['repo']);available={}
 for ref in list(refs.branches)+list(refs.tags):
  m=re.match(r'^stage1-step(\d+)-tokens',ref.name)
  if m:available[int(m[1])]=ref.name
 if len(available)<2:raise ValueError('No stage-1 checkpoint pairs discovered; provide OLMo2 repo with public stage1 refs')
 final=max(available);steps=sorted({min(available,key=lambda s:abs(s-final*f)) for f in cfg['training_fractions']})
 steps=[s for s in steps if s<final]
 if not steps:raise ValueError('Training fractions must select earlier checkpoints')
 result=dict(final=dict(repo=spec['repo'],revision=available[final],sha=api.model_info(spec['repo'],revision=available[final]).sha,step=final),
  earlier=[dict(repo=spec['repo'],revision=available[s],sha=api.model_info(spec['repo'],revision=available[s]).sha,step=s) for s in steps])
 dump(path,result);return result


def run(spec,rows,cfg,root,stop):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 refs=revisions(spec,cfg,root/'checkpoints.json')
 tok,final=model_load(refs['final'],cfg['device'])
 tokenizer_hash=digest(tok.backend_tokenizer.to_str());final_answers={}
 # Keep only CPU weight references; release final model before earlier GPU load.
 for r in rows:
  if stop.exists():return
  p=root/'final'/f'{key(r)}.json'
  if p.exists():result=load(p)
  else:
   ids,S,g=tokenize(tok,r,final.device)
   result=generate(final,tok,ids,r['aliases'],cfg['max_new_tokens']);dump(p,result)
  final_answers[key(r)]=result
 final_weights={k:m.weight.detach().cpu() for k,m in modules(final).items() if k.split(':')[1] in cfg['training_families']}
 del final;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
 for earlier in refs['earlier']:
  folder=root/f'step_{earlier["step"]}'
  if (folder/'DONE.json').exists():continue
  tok,e=model_load(earlier,cfg['device'])
  if digest(tok.backend_tokenizer.to_str())!=tokenizer_hash:raise ValueError('Tokenizer changed across checkpoints')
  mods=modules(e);deltas={};bases={}
  for name,W in final_weights.items():
   if stop.exists():return
   delta=W-mods[name].weight.detach().cpu();deltas[name]=delta
   q=min(cfg['training_rank']+cfg['svd_oversample'],*delta.shape)
   with torch.random.fork_rng():
    torch.manual_seed(cfg['seed']+int(digest(name)[:8],16));u,s,v=torch.svd_lowrank(delta,q=q,niter=cfg['svd_iterations'])
   rank=min(cfg['training_rank'],q,int((s>s[0]*1e-7).sum())) if float(s[0])>0 else 0
   randoms=[]
   for seed in cfg['seeds']:
    rng=torch.Generator().manual_seed(seed+int(digest(name)[:8],16))
    ru=torch.linalg.qr(torch.randn(delta.shape[0],rank,generator=rng),mode='reduced')[0]
    rv=torch.linalg.qr(torch.randn(delta.shape[1],rank,generator=rng),mode='reduced')[0];randoms.append((ru,rv))
   bases[name]=(u[:,:rank],v[:,:rank],randoms)
   dump(folder/'svd'/f'{name.replace(":","_")}.json',dict(rank=rank,singular_values=s.tolist(),delta_norm=float(delta.norm()),
    retained_energy=float(s[:rank].square().sum()/delta.square().sum().clamp_min(1e-30))))
  for i,r in enumerate(rows):
   if stop.exists():return
   path=folder/'facts'/f'{key(r)}.json'
   if path.exists():continue
   ids,S,g=tokenize(tok,r,e.device);C=e.capture(ids,cfg['tolerance']);other=C['z'].clone();other[g]=-torch.inf;c=int(other.argmax())
   old=generate(e,tok,ids,r['aliases'],cfg['max_new_tokens']);new=final_answers[key(r)]
   transition=('learned' if new['answer_correct'] else 'stayed_wrong') if not old['answer_correct'] else ('stayed_correct' if new['answer_correct'] else 'forgotten')
   _,L,F=split(e,C,g,c);measurements=[]
   for name,delta in deltas.items():
    U,V,randoms=bases[name];l,f=name.split(':')
    for channel,grad in [('live',L.get(name)),('content',F.get(name)),('selection',subtract(L.get(name),F.get(name)))]:
     from .core import dot
     energy=max(0.,dot(grad,grad));projection=projected_energy(grad,U,V)
     signed=delta_dot(grad,delta);dn=float(delta.double().norm())
     measurements.append(dict(layer=int(l),family=f,channel=channel,gradient_sq=energy,
      projection_fraction=projection/energy if energy>1e-20 else None,
      random_projection_fractions=[projected_energy(grad,ru,rv)/energy for ru,rv in randoms] if energy>1e-20 else [],
      signed_delta_dot=signed,delta_cosine=signed/(energy**.5*dn) if energy>1e-20 and dn>0 else None))
   dump(path,dict(prompt=r,baseline=old,final=new,transition=transition,g=g,c=c,certificate=C['certificate'],rows=measurements))
   print(f'Training {spec["key"]} step {earlier["step"]}: {i+1}/{len(rows)} facts',flush=True)
   del C,L,F
  dump(folder/'DONE.json',dict(status='complete'));del e,deltas,bases;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
