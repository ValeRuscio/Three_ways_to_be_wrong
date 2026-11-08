import gc,math
from pathlib import Path
import torch
from .core import *
from .io import dump,key,generate,digest


def split(e,C,g,c):
 objective=lambda z:z[g]-z[c]
 L,lv=factors(e,C,objective,'live');F,fv=factors(e,C,objective,'frozen')
 if abs(lv-fv)>1e-3*max(1,abs(lv)):raise AssertionError('Live/frozen scalar mismatch')
 rows=[]
 for name in modules(e):
  l,f=name.split(':');rows.append(dict(layer=int(l),family=f,**stats(L.get(name),F.get(name))))
 return rows,L,F


def fisher(e,C,draws,seed):
 # Conditional next-token model Fisher E_y~p [(d log p_y)(d log p_y)^T].
 rng=torch.Generator().manual_seed(seed)
 sampled=torch.multinomial(C['z'].double().softmax(-1).cpu(),draws,replacement=True,generator=rng)
 rows=[]
 for draw,y in enumerate(sampled.tolist()):
  obj=lambda z,y=y:z.log_softmax(-1)[y]
  L,_=factors(e,C,obj,'live');F,_=factors(e,C,obj,'frozen')
  for name in modules(e):
   l,f=name.split(':');rows.append(dict(draw=draw,sampled_token=y,layer=int(l),family=f,**stats(L.get(name),F.get(name))))
  del L,F
 return rows


def chosen_layers(e,cfg):
 specified=cfg.get('probe_layers')
 return sorted(set(max(0,min(e.L-1,round(f*(e.L-1)))) for f in cfg['probe_layer_fractions'])) if specified is None else specified


def seeds_for(r,cfg,stage):
 return [int(digest([cfg['seed'],key(r),stage,s])[:8],16) for s in cfg['seeds']]


@torch.no_grad()
def perturb(e,C,g,c,r,cfg,L,F,folder,stop=None):
 baseline=float(C['z'][g]-C['z'][c]);mods=modules(e)
 for l in chosen_layers(e,cfg):
  for family in cfg['families']:
   name=f'{l}:{family}';mod=mods[name];wn=float(mod.weight.norm())
   for seed in seeds_for(r,cfg,'perturb'):
    u,v=random_one(mod.weight.shape,seed+l*101+list(FAMILIES).index(family))
    dl=directional(L.get(name),u,v);df=directional(F.get(name),u,v)
    for dose in cfg['doses']:
     if stop is not None and stop.exists():return
     path=Path(folder)/(digest([name,seed,dose])[:24]+'.json')
     if path.exists():continue
     alpha=dose*wn
     with patch(mod,u,v,alpha):
      zl=live(e,C['ids']);zf=frozen(e,C)
     a=float(zl[g]-zl[c])-baseline;b=float(zf[g]-zf[c])-baseline
     # With one untied matrix changed and program fixed, response is affine in W.
     linerr=abs(b-alpha*df)/max(1.,abs(b),abs(alpha*df))
     if linerr>cfg['finite_tolerance']:raise AssertionError(f'Frozen finite-edit certificate failed {name}: {linerr}')
     dump(path,dict(layer=l,family=family,seed=seed,dose=dose,delta_norm=alpha,
      live_change=a,frozen_change=b,selection_response=a-b,abs_selection_response=abs(a-b),
      abs_live_nonlinear_residual=abs(a-alpha*dl),relative_live_frozen_gap=abs(a-b)/max(1.,abs(a),abs(b)),
      first_order_live=alpha*dl,first_order_content=alpha*df,first_order_selection=alpha*(dl-df),
      live_nonlinear_residual=a-alpha*dl,frozen_affinity_error=linerr,
      canonical_top1=bool(zl.argmax()==g)))


@torch.no_grad()
def frontier(e,tok,C,g,c,r,cfg,L,F,controls,folder,stop=None):
 baseline=float(C['z'][g]-C['z'][c]);mods=modules(e)
 for l in chosen_layers(e,cfg):
  for family in cfg['frontier_families']:
   name=f'{l}:{family}';mod=mods[name];wn=float(mod.weight.norm())
   directions={'live':rank_one(L.get(name)),'content':rank_one(F.get(name)),
    'selection':rank_one(subtract(L.get(name),F.get(name)))}
   directions.update({f'random_{seed}':random_one(mod.weight.shape,seed+l*101+list(FAMILIES).index(family)) for seed in seeds_for(r,cfg,'frontier')})
   for channel,direction in directions.items():
    for dose in cfg['doses']:
     if stop is not None and stop.exists():return
     path=Path(folder)/(digest([name,channel,dose])[:24]+'.json')
     if path.exists():continue
     if direction is None:
      dump(path,dict(layer=l,family=family,channel=channel,dose=dose,status='zero_gradient'));continue
     u,v=direction;alpha=dose*wn
     with patch(mod,u,v,alpha):
      z=live(e,C['ids']);answer=generate(e,tok,C['ids'],r['aliases'],cfg['max_new_tokens']);probes=[]
      for q in controls:
       new=live(e,q['ids']);old=q['z'].to(new)
       kl=float((old.double().softmax(-1)*(old.double().log_softmax(-1)-new.double().log_softmax(-1))).sum())
       gen=generate(e,tok,q['ids'],q['aliases'],cfg['max_new_tokens']) if cfg['control_generation'] else {}
       probes.append(dict(id=q['id'],group_id=q['group_id'],kl=kl,prediction_changed=bool(old.argmax()!=new.argmax()),
        baseline_answer_correct=q['answer_correct'],**gen))
     dump(path,dict(layer=l,family=family,channel=channel,dose=dose,delta_norm=alpha,status='measured',
      margin_gain=float(z[g]-z[c])-baseline,canonical_top1=bool(z.argmax()==g),**answer,controls=probes))


@torch.no_grad()
def weight_bases(e,cfg):
 out={}
 for l,b in enumerate(e.blocks):
  W=b.mlp.gate_proj.weight.detach().cpu();q=min(cfg['geometry_rank']+cfg['svd_oversample'],*W.shape)
  with torch.random.fork_rng():
   torch.manual_seed(cfg['seed']+l);u,s,v=torch.svd_lowrank(W,q=q,niter=cfg['svd_iterations'])
  rank=min(cfg['geometry_rank'],q);res=float((W@v[:,:rank]-u[:,:rank]*s[:rank]).norm()/W.norm().clamp_min(1e-20))
  out[l]=dict(V=v[:,:rank],singular_values=s[:rank],left_residual=res)
 return out


def angles(X,V,rank):
 # Uncentered prompt-origin span; explicit rank/cutoff avoids zero singular vectors.
 _,s,vh=torch.linalg.svd(X.double(),full_matrices=False)
 if not len(s) or s[0]<=1e-14:return dict(rank=0,cosines=[],mean_cosine_squared=None)
 r=min(rank,int((s>s[0]*1e-8).sum()));Q=vh[:r].T
 vals=torch.linalg.svdvals(V.double().T@Q).clamp(0,1)
 return dict(rank=r,cosines=vals.tolist(),mean_cosine_squared=float(vals.square().mean()))


def geometry_report(paths,bases,cfg):
 arrays=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
 if not arrays:return []
 X=torch.stack(arrays);rows=[]
 for l,base in bases.items():
  for i,name in enumerate(['subject','template']):
   U=base['V'];a=angles(X[:,l,i],U,cfg['geometry_rank'])
   vals=[]
   for seed in cfg['seeds']:
    rng=torch.Generator().manual_seed(seed+l*101)
    Q=torch.linalg.qr(torch.randn(U.shape,generator=rng),mode='reduced')[0]
    vals.append(angles(X[:,l,i],Q,cfg['geometry_rank'])['mean_cosine_squared'])
   rows.append(dict(layer=l,origin=name,prompts=len(paths),svd_residual=base['left_residual'],
    weight_singular_values=base['singular_values'].tolist(),random_mean_cosine_squared=vals,**a))
 return rows
