"""Exact local MLP-edge factors under the native pass's frozen program.
A per-neuron edge contribution is not a disjoint partition across layers.
"""
import torch
import numpy as np
from causal_suite.engine import norm

SITES=['subject','answer']


def require_swiglu(engine):
 if engine.kind not in ['llama','qwen2','olmo2']:raise ValueError('This study requires a SiLU-gated MLP: Llama/Qwen2/OLMo2. GPT-NeoX has no separate gate_proj.')
 if getattr(engine.model.config,'hidden_act','silu') not in ['silu','swish']:raise ValueError('Only SiLU gates are supported')


def mlp_input(engine,C,l,x):
 captured=[]
 def hook(module,args):captured.append(args[0])
 h=engine.blocks[l].mlp.up_proj.register_forward_pre_hook(hook)
 try:engine.layer(x,l,C)
 finally:h.remove()
 return captured[0]


def output_pullback(engine,C,l,r):
 if engine.kind!='olmo2':return r
 b=engine.blocks[l];d=C['layers'][l]
 # OLMo post-MLP RMSNorm is between W_down and block output.
 return r*d['post_feedforward_layernorm']*b.post_feedforward_layernorm.weight


@torch.no_grad()
def extract(engine,C,S,g,c,tol=1e-4):
 require_swiglu(engine);positions=[S[-1],len(C['ids'])-1];states=engine.component(C,S)
 rg=engine.pullbacks(C,g);rc=engine.pullbacks(C,c);features={k:[] for k in ['pre','gate','sigmoid','up_native','up_subject','down_gold','down_competitor','contribution_gold','contribution_margin','gate_row_norm','up_cosine','down_cosine','signed_distance','input_norm']}
 certificates=[]
 for l,b in enumerate(engine.blocks):
  native=mlp_input(engine,C,l,C['layers'][l]['x']);zero=torch.zeros_like(states[l]);subject=mlp_input(engine,C,l,states[l])-mlp_input(engine,C,l,zero)
  pre=b.mlp.gate_proj(native)[positions];gate=C['layers'][l]['gate'][positions];up=b.mlp.up_proj(native)[positions]
  us=(subject@b.mlp.up_proj.weight.T)[positions]
  pg=output_pullback(engine,C,l,rg[l+1])[positions];pc=output_pullback(engine,C,l,rc[l+1])[positions]
  dg=pg@b.mlp.down_proj.weight;dc=pc@b.mlp.down_proj.weight
  contribution=gate*us*dg;cm=gate*us*(dg-dc)
  direct=(gate*us)@b.mlp.down_proj.weight.T
  err=float(((contribution.sum(-1)-(direct*pg).sum(-1)).abs()/((direct*pg).sum(-1).abs().clamp_min(1))).max())
  if err>tol:raise AssertionError(f'Local three-factor certificate failed: {err}')
  certificates.append(err)
  gn=b.mlp.gate_proj.weight.norm(dim=1)[None].expand(len(positions),-1);un=b.mlp.up_proj.weight.norm(dim=1)[None]
  sn=subject[positions].norm(dim=-1,keepdim=True);dn=b.mlp.down_proj.weight.norm(dim=0)[None];pn=pg.norm(dim=-1,keepdim=True)
  vals=dict(pre=pre,gate=gate,sigmoid=pre.sigmoid(),up_native=up,up_subject=us,down_gold=dg,down_competitor=dc,
   contribution_gold=contribution,contribution_margin=cm,gate_row_norm=gn,
   up_cosine=us/(un*sn).clamp_min(1e-20),down_cosine=dg/(dn*pn).clamp_min(1e-20),
   signed_distance=pre/gn.clamp_min(1e-20),input_norm=native[positions].norm(dim=-1,keepdim=True).expand_as(pre))
  for k,v in vals.items():features[k].append(v.cpu().numpy().astype(np.float32))
 features={k:np.stack(v) for k,v in features.items()}
 # Independently selected direct subject-position -> answer hand-off heads.
 features['head_gold']=engine.head_scores(C,S,g).cpu().numpy()
 features['head_margin']=(engine.head_scores(C,S,g)-engine.head_scores(C,S,c)).cpu().numpy()
 features['factor_certificate']=np.array(certificates)
 features['attention_rows']=torch.stack([d['A'][:,-1] for d in C['layers']]).cpu().numpy()
 features['attention_roles']=role_mass(features['attention_rows'],S)
 return features


def role_indices(T,S):
 s=set(S);first=[] if 0 in s else [0]
 return [first,list(S),[i for i in range(T) if i not in s and i!=0 and i<min(S)], [i for i in range(T) if i not in s and i!=0 and i>=min(S)]]

def role_mass(rows,S):return np.stack([rows[...,ix].sum(-1) for ix in role_indices(rows.shape[-1],S)],axis=-1)


def native_value_direction(engine,ids,layer,position,g,c):
 """True live-network margin derivative at W_down output, not frozen r.
 This is used only to orient a rank-one value-edit baseline.
 """
 captured=[]
 def hook(module,args,out):
  y=out.detach().requires_grad_(True);captured.append(y);return y
 h=engine.blocks[layer].mlp.down_proj.register_forward_hook(hook)
 try:
  with torch.enable_grad():
   z=engine.model(ids[None],attention_mask=torch.ones_like(ids[None]),use_cache=False).logits[0,-1]
   grad=torch.autograd.grad(z[g]-z[c],captured[0])[0]
 finally:h.remove()
 return grad[0,position].detach()
