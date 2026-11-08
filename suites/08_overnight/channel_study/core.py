"""Exact token-factor representation of parameter gradients: G = D.T @ X.
Weights remain fixed; autograd follows the embedding input. Linear output hooks
recover the weight derivative at every use without allocating dense gradients.
"""
import contextlib, math
import torch
from causal_suite.engine import norm

FAMILIES = {'gate': ('mlp','gate_proj'), 'up': ('mlp','up_proj'),
 'down': ('mlp','down_proj'), 'q': ('self_attn','q_proj'),
 'k': ('self_attn','k_proj'), 'v': ('self_attn','v_proj'),
 'o': ('self_attn','o_proj')}


def modules(e):
 if e.kind not in ('llama','qwen2','olmo2'):raise ValueError('Only Llama, Qwen2, OLMo2 are supported')
 if getattr(e.model.config,'hidden_act','silu') not in ('silu','swish'):raise ValueError('Requires SiLU MLP')
 return {f'{l}:{f}':getattr(getattr(b,a),p) for l,b in enumerate(e.blocks) for f,(a,p) in FAMILIES.items()}


def frozen(e,C):
 x=C['layers'][0]['x'].detach().requires_grad_(True)
 for l in range(e.L):x=e.layer(x,l,C)
 return e.read(x,C)


def live(e,ids):
 return e.model(ids[None],attention_mask=torch.ones_like(ids[None]),use_cache=False).logits[0,-1]


def factors(e,C,objective,mode):
 collected={};handles=[]
 def hook(key):
  def forward(mod,args,out):
   x=args[0].detach().reshape(-1,args[0].shape[-1]).cpu()
   if not out.requires_grad:return
   def backward(delta):
    d=delta.detach().reshape(-1,delta.shape[-1]).cpu()
    collected.setdefault(key,[]).append((d,x))
   out.register_hook(backward)
  return forward
 try:
  for key,mod in modules(e).items():handles.append(mod.register_forward_hook(hook(key)))
  if mode=='live':
   def emb_hook(mod,args,out):return out.detach().requires_grad_(True)
   handles.append(e.model.get_input_embeddings().register_forward_hook(emb_hook))
   z=live(e,C['ids'])
  else:z=frozen(e,C)
  loss=objective(z);loss.backward()
  value=float(loss.detach());del loss,z
 finally:
  for h in handles:h.remove()
 return {key:(torch.cat([p[0] for p in pairs]),torch.cat([p[1] for p in pairs])) for key,pairs in collected.items()},value


def dot(a,b):
 if a is None or b is None:return 0.
 d,x=a;e,y=b
 return float(((d.double()@e.double().T)*(x.double()@y.double().T)).sum())


def subtract(a,b):
 if a is None:return None if b is None else (-b[0],b[1])
 if b is None:return a
 return torch.cat((a[0],-b[0])),torch.cat((a[1],b[1]))


def stats(a,b):
 # live=a, content=b; selection=a-b. Energies are not additive without cross.
 aa=max(0.,dot(a,a));bb=max(0.,dot(b,b));ab=dot(a,b)
 ss=max(0.,aa+bb-2*ab);cross=ab-bb
 cn=math.sqrt(bb);sn=math.sqrt(ss)
 return dict(live_sq=aa,content_sq=bb,selection_sq=ss,cross= cross,
  content_norm=cn,selection_norm=sn,live_norm=math.sqrt(aa),
  selection_fraction=sn/(sn+cn) if sn+cn>1e-15 else None,
  content_selection_cosine=max(-1.,min(1.,cross/(cn*sn))) if cn*sn>1e-15 else None,
  closure_error=abs(aa-(bb+ss+2*cross))/max(1.,aa))


def directional(g,u,v):
 if g is None:return 0.
 d,x=g
 return float(((d.double()@u.double())*(x.double()@v.double())).sum())


def rank_one(g):
 if g is None:return None
 d,x=g
 # Exact thin QR + small SVD; no dense parameter-gradient allocation.
 qd,rd=torch.linalg.qr(d.double().T,mode='reduced')
 qx,rx=torch.linalg.qr(x.double().T,mode='reduced')
 u,s,vh=torch.linalg.svd(rd@rx.T,full_matrices=False)
 if not len(s) or s[0]<1e-15:return None
 return (qd@u[:,0]).float(),(qx@vh[0]).float()


def random_one(shape,seed):
 rng=torch.Generator().manual_seed(seed)
 u=torch.randn(shape[0],generator=rng);v=torch.randn(shape[1],generator=rng)
 return u/u.norm(),v/v.norm()


@contextlib.contextmanager
def patch(mod,u,v,alpha):
 # Functional W + alpha*u*v.T; applies at every sequence position, including generation.
 u=u.to(mod.weight);v=v.to(mod.weight)
 def change(m,args,out):return out+alpha*(args[0]@v).unsqueeze(-1)*u
 h=mod.register_forward_hook(change)
 try:yield
 finally:h.remove()


def mlp_input(e,C,l,x):
 box=[]
 h=e.blocks[l].mlp.up_proj.register_forward_pre_hook(lambda m,a:box.append(a[0]))
 try:e.layer(x,l,C)
 finally:h.remove()
 return box[0]


@torch.no_grad()
def origins(e,C,S,tol):
 x0=C['layers'][0]['x'];groups={'subject':S,'first':[] if 0 in S else [0],
  'template':[i for i in range(len(x0)) if i not in S and i!=0]}
 states={}
 for name,positions in groups.items():
  x=torch.zeros_like(x0);x[positions]=x0[positions];states[name]=x
 bias=torch.zeros_like(x0);rows=[];vectors=[]
 for l,b in enumerate(e.blocks):
  zero=torch.zeros_like(x0);mi0=mlp_input(e,C,l,zero)
  inp={k:mlp_input(e,C,l,x)-mi0 for k,x in states.items()}
  ib=mlp_input(e,C,l,bias)
  native=mlp_input(e,C,l,C['layers'][l]['x'])
  actual=b.mlp.gate_proj(native)[-1]
  pre={k:b.mlp.gate_proj.weight@x[-1] for k,x in inp.items()}
  pb=b.mlp.gate_proj(ib)[-1]
  err=float((sum(pre.values())+pb-actual).norm()/actual.norm().clamp_min(1))
  if err>tol:raise AssertionError(f'Gate origin certificate {err}')
  rows.append(dict(layer=l,certificate=err,subject_norm=float(pre['subject'].norm()),
   template_norm=float(pre['template'].norm()),first_norm=float(pre['first'].norm()),bias_norm=float(pb.norm()),
   subject_template_dot=float(pre['subject']@pre['template']),
   subject_exceeds_template=bool(pre['subject'].norm()>pre['template'].norm())))
  vectors.append(torch.stack([inp['subject'][-1],inp['template'][-1]]).cpu())
  states={k:e.linear(x,l,C) for k,x in states.items()};bias=e.layer(bias,l,C)
 return rows,torch.stack(vectors)


def projected_energy(g,U,V):
 if g is None:return 0.
 d,x=g
 return float(((d.double()@U.double()).T@(x.double()@V.double())).square().sum())


def delta_dot(g,delta):
 if g is None:return 0.
 d,x=g
 # Float64 accumulation, CPU; full displacement, not only its low-rank approximation.
 return float((d.double()*(delta.double()@x.double().T).T).sum())
