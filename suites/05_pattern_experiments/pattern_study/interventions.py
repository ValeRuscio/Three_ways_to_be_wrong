from contextlib import contextmanager
import numpy as np
import torch
from gate_study.factors import role_indices
from .common import *


@contextmanager
def gates(e,changes,position):
 """changes[layer]=(indices, replacement ACTUAL SiLU gate values)."""
 hs=[];ups={}
 try:
  for l,(ix,values) in changes.items():
   b=e.blocks[l];ix=torch.as_tensor(ix,device=e.device);values=torch.as_tensor(values,device=e.device,dtype=torch.float32)
   def up(mod,args,out,l=l):ups[l]=out
   def pre(mod,args,l=l,ix=ix,values=values):
    x=args[0].clone();x[:,position,ix]=ups[l][:,position,ix]*values
    return (x,*args[1:])
   hs.append(b.mlp.up_proj.register_forward_hook(up));hs.append(b.mlp.down_proj.register_forward_pre_hook(pre))
  yield
 finally:
  for h in hs:h.remove()


@contextmanager
def attention(e,rows):
 """Full valid replacement last-prompt query rows; live recipient V and downstream."""
 hs=[]
 try:
  for l,row in rows.items():
   b=e.blocks[l];row=torch.as_tensor(row,device=e.device,dtype=torch.float32)
   def hook(mod,args,kw,out,l=l,b=b,row=row):
    x=(args[0] if args else kw['hidden_states'])[0];A=out[1][0].clone();A[:,-1]=row
    if (A<0).any() or not torch.allclose(A.sum(-1),torch.ones_like(A.sum(-1)),atol=1e-5):raise AssertionError('Invalid probability row')
    y=b.self_attn.o_proj((A@e.values(x,l)).transpose(0,1).reshape(len(x),-1))[None]
    result=list(out);result[0]=y;result[1]=A[None];return tuple(result)
   hs.append(b.self_attn.register_forward_hook(hook,with_kwargs=True))
  yield
 finally:
  for h in hs:h.remove()


def frozen(e,C,gate_changes=None,row_changes=None,position=-1):
 new=dict(C,layers=[dict(d) for d in C['layers']])
 for l,(ix,values) in (gate_changes or {}).items():
  g=new['layers'][l]['gate'].clone();g[position,torch.as_tensor(ix,device=e.device)]=torch.as_tensor(values,device=e.device,dtype=torch.float32);new['layers'][l]['gate']=g
 for l,row in (row_changes or {}).items():
  A=new['layers'][l]['A'].clone();A[:,-1]=torch.as_tensor(row,device=e.device,dtype=torch.float32);new['layers'][l]['A']=A
 with torch.no_grad():
  x=C['layers'][0]['x']
  for l in range(e.L):x=e.layer(x,l,new)
  return e.read(x,new)


def align_rows(donor,source,Sd,Sr):
 """Mass-conserving monotone transport within explicit semantic roles.
 Not raw token correspondence and not an isolated query/key intervention.
 Donor bins and recipient bins partition [0,1] uniformly within each role.
 """
 out=np.zeros_like(source)
 for di,ri in zip(role_indices(donor.shape[-1],Sd),role_indices(source.shape[-1],Sr)):
  if not ri:
   if di and np.max(donor[...,di].sum(-1))>1e-7:raise ValueError('Donor mass in a role missing from recipient')
   continue
  if not di:continue
  a=np.arange(len(di))/len(di);b=(np.arange(len(di))+1)/len(di)
  c=np.arange(len(ri))/len(ri);d=(np.arange(len(ri))+1)/len(ri)
  M=np.maximum(0,np.minimum(b[:,None],d)-np.maximum(a[:,None],c))*len(di)
  out[...,ri]=donor[...,di]@M
 if not np.allclose(out.sum(-1),1,atol=1e-5):raise AssertionError('Alignment did not conserve mass')
 return out


def edge_gradient(e,C,g,c):
 """Exact derivatives of frozen-program fixed margin wrt native A entries."""
 new=dict(C,layers=[dict(d,A=d['A'].detach().requires_grad_(True)) for d in C['layers']])
 with torch.enable_grad():
  x=C['layers'][0]['x']
  for l in range(e.L):x=e.layer(x,l,new)
  z=e.read(x,new,[g,c]);gr=torch.autograd.grad(z[0]-z[1],[d['A'] for d in new['layers']])
 return torch.stack([v[:,-1].detach() for v in gr]).cpu().numpy()


def interpolate_edges(base,donor,chosen):
 # Replace selected probabilities then normalize each row. Scores used to order
 # edits are local diagnostics; measured finite effects include renormalization.
 x=base.copy();x.reshape(-1)[chosen]=donor.reshape(-1)[chosen]
 sums=x.sum(-1,keepdims=True)
 if np.any(sums<=0):raise ValueError('Empty attention row after edit')
 return x/sums
