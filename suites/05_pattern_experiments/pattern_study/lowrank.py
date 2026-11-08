"""Exact low-rank functional weight deltas and live full-token gradients."""
from contextlib import contextmanager
import torch


def compress(U,V,rank):
 """Delta = U @ V.T; deterministic thin QR/SVD; no giant weight SVD."""
 U=U.double();V=V.double();qu,ru=torch.linalg.qr(U,mode='reduced');qv,rv=torch.linalg.qr(V,mode='reduced')
 a,s,b=torch.linalg.svd(ru@rv.T,full_matrices=False)
 if len(s)==0 or float(s[0])<=1e-20:return U[:,:0].float(),V[:,:0].float()
 k=min(rank,int((s>s[0]*1e-7).sum()))
 return ((qu@a[:,:k])*s[:k].sqrt()).float(),((qv@b[:k].T)*s[:k].sqrt()).float()


def frob(U,V):return float(torch.sqrt(((U.double().T@U.double())*(V.double().T@V.double())).sum().clamp_min(0)))


def fit(X,D,rank,ridge):
 """X [input,N], D [output,N]. Ridge-LS delta then rank truncation.
 Exact ridge solution before truncation; truncation is in Frobenius norm,
 not falsely called an optimal reduced-rank regression solution.
 """
 X=X.double();D=D.double();gram=X.T@X;lam=ridge*float(gram.diag().mean().clamp_min(1e-12))
 V=X@torch.linalg.solve(gram+lam*torch.eye(len(gram),dtype=X.dtype,device=X.device),torch.eye(len(gram),dtype=X.dtype,device=X.device))
 return compress(D,V,rank)


def random_delta(outdim,indim,rank,generator):
 r=min(rank,outdim,indim)
 u=torch.linalg.qr(torch.randn(outdim,r,generator=generator),mode='reduced')[0]
 v=torch.linalg.qr(torch.randn(indim,r,generator=generator),mode='reduced')[0]
 return u,v


def normalize(blocks,budget):
 n=sum(frob(u,v)**2 for u,v in blocks.values())**.5
 if n<=1e-20:return None
 return {l:(u*(budget/n),v) for l,(u,v) in blocks.items()}


@contextmanager
def apply(e,kind,blocks):
 """W x + U(V^T x), applied to all tokens, including generated tokens.
 This is a real additive weight intervention without copying dense weights.
 Hooks are removed even on exceptions. Delta norm is parameter Frobenius norm.
 """
 handles=[]
 try:
  for l,(u,v) in blocks.items():
   module=getattr(e.blocks[l].mlp,kind+'_proj');u=u.to(e.device);v=v.to(e.device)
   def hook(module,args,out,u=u,v=v):return out+(args[0]@v)@u.T
   handles.append(module.register_forward_hook(hook))
  yield
 finally:
  for h in handles:h.remove()


def gradients(e,m,kind,ls):
 """Full live derivative, including all token positions, represented as G X^T.
 All three weight families use the SAME recipient fixed-competitor margin.
 """
 handles=[];observed={}
 emb=e.model.get_input_embeddings()
 handles.append(emb.register_forward_hook(lambda mod,args,out:out.detach().requires_grad_(True)))
 try:
  for l in ls:
   def hook(mod,args,out,l=l):observed[l]=(args[0],out)
   handles.append(getattr(e.blocks[l].mlp,kind+'_proj').register_forward_hook(hook))
  x=torch.tensor(m['input_ids'],device=e.device)
  with torch.enable_grad():
   z=e.model(x[None],attention_mask=torch.ones_like(x[None]),use_cache=False).logits[0,-1]
   outs=[observed[l][1] for l in ls]
   gs=torch.autograd.grad(z[m['g']]-z[m['c']],outs)
  return {l:(g[0].detach().T.cpu(),observed[l][0][0].detach().T.cpu()) for l,g in zip(ls,gs)}
 finally:
  for h in handles:h.remove()
