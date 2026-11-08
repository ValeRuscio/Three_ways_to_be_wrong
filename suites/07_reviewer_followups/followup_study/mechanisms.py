"""Conditional native replay. Freeze selected realized mechanisms, not weights.
All sites are frozen, including attention Q/K RMSNorms when present.
"""
from contextlib import contextmanager
from itertools import combinations
import math
import torch
from causal_suite.engine import norm
from channel_study.core import modules,dot,subtract,stats,random_one,rank_one,directional,patch,live,frozen

MECHANISMS=('attention','gate','norm')

def norm_modules(e):
 return {name:mod for name,mod in e.model.named_modules()
         if isinstance(mod,torch.nn.LayerNorm) or 'rmsnorm' in mod.__class__.__name__.lower()}


@torch.no_grad()
def capture(e,ids,tol):
 scales={};handles=[]
 for name,mod in norm_modules(e).items():
  def save(mod,args,name=name):
   x=args[0].detach();y=x-x.mean(-1,keepdim=True) if isinstance(mod,torch.nn.LayerNorm) else x
   eps=getattr(mod,'variance_epsilon',getattr(mod,'eps',1e-5))
   scales[name]=torch.rsqrt(y.float().square().mean(-1,keepdim=True)+eps)
  handles.append(mod.register_forward_pre_hook(save))
 try:
  # e.capture also runs frozen replay internally. Save native norm inputs only;
  # engine.norm is functional, so native norm module hooks are not called again.
  C=e.capture(ids,tol)
 finally:
  for h in handles:h.remove()
 C['norm_scales']=scales
 return C


@contextmanager
def freezes(e,C,which=(),gate_layers=None):
 which=set(which);handles=[]
 try:
  if 'norm' in which:
   for name,mod in norm_modules(e).items():
    inv=C['norm_scales'][name]
    def replace(mod,args,out,inv=inv):return norm(args[0],mod,inv)
    handles.append(mod.register_forward_hook(replace))
  for l,b in enumerate(e.blocks):
   if 'gate' in which and (gate_layers is None or l in gate_layers):
    def gate(mod,args,out,l=l):return C['layers'][l]['gate'][None].expand_as(out)
    if not isinstance(b.mlp.act_fn,torch.nn.Module):raise ValueError('Expected hookable SiLU module')
    handles.append(b.mlp.act_fn.register_forward_hook(gate))
   if 'attention' in which:
    def att(mod,args,kw,out,l=l,b=b):
     x=(args[0] if args else kw['hidden_states'])[0]
     A=C['layers'][l]['A'];v=e.values(x,l)
     y=b.self_attn.o_proj((A@v).transpose(0,1).reshape(len(x),-1))[None]
     result=list(out);result[0]=y
     if len(result)>1:result[1]=A[None]
     return tuple(result)
    handles.append(b.self_attn.register_forward_hook(att,with_kwargs=True))
  yield
 finally:
  for h in handles:h.remove()


def collect(e,C,objective,which=(),gate_layers=None,readout=None,selected=None):
 """Exact parameter gradient factors at all token uses; optional hidden readout."""
 records={};handles=[];hidden=[]
 def proj(key):
  def forward(mod,args,out):
   if not out.requires_grad:return
   x=args[0].detach().reshape(-1,args[0].shape[-1]).cpu()
   def backward(d):records.setdefault(key,[]).append((d.detach().reshape(-1,d.shape[-1]).cpu(),x))
   out.register_hook(backward)
  return forward
 # Freeze hooks first, so final-norm readout hook observes the replaced tensor.
 with freezes(e,C,which,gate_layers):
  try:
   handles.append(e.model.get_input_embeddings().register_forward_hook(lambda m,a,out:out.detach().requires_grad_(True)))
   for key,mod in modules(e).items():
    if selected is None or key in selected:handles.append(mod.register_forward_hook(proj(key)))
   if readout is not None:handles.append(e.final.register_forward_hook(lambda m,a,out:hidden.append(out[0,-1])))
   z=live(e,C['ids']);loss=objective(z) if readout is None else hidden[-1]@readout.to(hidden[-1])
   if loss.requires_grad:loss.backward()
   value=float(loss.detach())
  finally:
   for h in handles:h.remove()
 return {k:(torch.cat([d for d,x in v]),torch.cat([x for d,x in v])) for k,v in records.items()},value


def combine(terms):
 items=[(weight,g) for weight,g in terms if g is not None and weight!=0]
 if not items:return None
 return torch.cat([weight*g[0] for weight,g in items]),torch.cat([g[1] for weight,g in items])


def subsets():
 return [frozenset(c) for n in range(4) for c in combinations(MECHANISMS,n)]


def split_rows(e,L,F):
 rows=[]
 for name in modules(e):
  l,f=name.split(':');rows.append(dict(layer=int(l),family=f,**stats(L.get(name),F.get(name))))
 return rows


def three_way(e,C,g,c):
 obj=lambda z:z[g]-z[c];G={};values={};masks=subsets();indices={m:i for i,m in enumerate(masks)}
 for mask in masks:G[mask],values['+'.join(sorted(mask)) or 'live']=collect(e,C,obj,mask)
 discrepancy=max(values.values())-min(values.values())
 if discrepancy>1e-3*max(1.,max(abs(v) for v in values.values())):raise AssertionError('Partial freezes changed baseline objective')
 empty=frozenset();full=frozenset(MECHANISMS);rows=[]
 selection=torch.zeros(8,dtype=torch.float64);selection[indices[empty]]=1;selection[indices[full]]=-1
 coefficients={};marginals={}
 for mech in MECHANISMS:
  v=torch.zeros(8,dtype=torch.float64);marginals[mech]=v.clone();marginals[mech][indices[empty]]=1;marginals[mech][indices[frozenset([mech])]]=-1
  for mask in masks:
   if mech in mask:continue
   w=math.factorial(len(mask))*math.factorial(2-len(mask))/math.factorial(3)
   v[indices[mask]]+=w;v[indices[mask|{mech}]]-=w
  coefficients[mech]=v
 for name in modules(e):
  # Gram of eight token-factor gradients. Reuse its 36 inner products instead
  # of materializing long concatenated Shapley factors for every statistic.
  K=torch.zeros((8,8),dtype=torch.float64)
  for i,mask in enumerate(masks):
   for j in range(i+1):K[i,j]=K[j,i]=dot(G[mask].get(name),G[masks[j]].get(name))
  ip=lambda a,b:float(a@K@b)
  energy=max(0.,ip(selection,selection));residual=sum(coefficients.values())-selection
  error=math.sqrt(max(0.,ip(residual,residual)))/max(1.,math.sqrt(energy))
  l,f=name.split(':');row=dict(layer=int(l),family=f,selection_sq=energy,closure=error)
  for mech in MECHANISMS:
   P=coefficients[mech];M=marginals[mech]
   row[mech+'_norm']=math.sqrt(max(0.,ip(P,P)))
   row[mech+'_signed_projection_share']=ip(P,selection)/energy if energy>1e-20 else None
   row[mech+'_one_freeze_norm']=math.sqrt(max(0.,ip(M,M)))
  for a,b in combinations(MECHANISMS,2):row[a+'_'+b+'_cross']=ip(coefficients[a],coefficients[b])
  rows.append(row)
 return dict(rows=rows,objective_values=values,max_baseline_disagreement=discrepancy)


def certificates(e,C,g,c,cfg):
 obj=lambda z:z[g]-z[c];L,lv=collect(e,C,obj);F,fv=collect(e,C,obj,MECHANISMS);rows=[]
 for name,mod in modules(e).items():
  if int(name.split(':')[0]) not in cfg['certificate_layers_resolved']:continue
  directions=[('random',seed,random_one(mod.weight.shape,seed+int(name.split(':')[0])*31+list(modules(e)).index(name))) for seed in cfg['seeds']]
  aligned=rank_one(L.get(name))
  if aligned is not None:directions.append(('gradient_aligned',-1,aligned))
  for direction,seed,(u,v) in directions:
   wn=float(mod.weight.norm());pl=directional(L.get(name),u,v);pf=directional(F.get(name),u,v)
   for epsilon in cfg['fd_epsilons']:
    alpha=epsilon*wn
    with torch.no_grad():
     with patch(mod,u,v,alpha):
      zp=live(e,C['ids']);fp=frozen(e,C)
     with patch(mod,u,v,-alpha):zm=live(e,C['ids'])
    measured=float(obj(zp)-obj(zm));predicted=2*alpha*pl
    frozen_delta=float(obj(fp))-fv;frozen_pred=alpha*pf
    absolute=abs(measured-predicted)
    # Certify changes, not division by a possibly near-zero slope.
    limit=cfg['fd_atol']+cfg['fd_rtol']*max(abs(measured),abs(predicted))
    flimit=cfg['fd_atol']+cfg['frozen_rtol']*max(abs(frozen_delta),abs(frozen_pred))
    l,f=name.split(':');rows.append(dict(layer=int(l),family=f,direction=direction,seed=seed,epsilon=epsilon,delta_norm=alpha,weight_norm=wn,
     live_predicted_difference=predicted,live_measured_difference=measured,live_absolute_error=absolute,
     live_pass=absolute<=limit,live_resolved=abs(predicted)>10*cfg['fd_atol'],
     frozen_predicted_change=frozen_pred,frozen_actual_change=frozen_delta,
     frozen_absolute_error=abs(frozen_delta-frozen_pred),frozen_pass=abs(frozen_delta-frozen_pred)<=flimit))
 return dict(rows=rows,baseline_disagreement=abs(lv-fv),native_certificate=C['certificate'])


def layer_matrix(e,C,g,c,cfg,stop=None):
 names=[n for n in modules(e) if n.split(':')[1] in cfg['matrix_families']]
 obj=lambda z:z[g]-z[c];L,_=collect(e,C,obj,selected=names);F,_=collect(e,C,obj,MECHANISMS,selected=names);rows=[]
 for j in range(e.L):
  if stop is not None and stop.exists():return None
  H,_=collect(e,C,obj,['gate'],gate_layers={j},selected=names)
  if (j+1)%4==0 or j+1==e.L:print(f'Gate-layer matrix: {j+1}/{e.L} downstream layers measured',flush=True)
  for i in range(e.L):
   total=effect=alignment=0.
   for name in names:
    if int(name.split(':')[0])!=i:continue
    S=subtract(L.get(name),F.get(name));D=subtract(L.get(name),H.get(name))
    total+=max(0.,dot(S,S));effect+=max(0.,dot(D,D));alignment+=dot(D,S)
   rows.append(dict(source_layer=i,frozen_gate_layer=j,strict_downstream=j>i,effect_norm=math.sqrt(effect),
    selection_norm=math.sqrt(total),relative_effect=math.sqrt(effect/total) if total>1e-20 else None,
    signed_selection_alignment=alignment/total if total>1e-20 else None))
 return dict(rows=rows,interpretation='One-layer gate freeze marginal effects overlap; columns are not an additive partition.')
