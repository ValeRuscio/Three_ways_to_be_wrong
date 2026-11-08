"""Per-head output-coordinate gradient geometry; never substitutes fresh gradients
for unavailable pretraining history. Local writes overlap across computational
paths and must NOT be summed as disjoint input origins.
"""
from pathlib import Path
import json
import numpy as np
import torch
from causal_suite.engine import norm
from .common import dump


def load_manifest(path,model_sha,width):
 if not path:return None
 p=Path(path);m=json.loads(p.read_text())
 if m['model_sha']!=model_sha:raise ValueError('Gradient basis checkpoint mismatch')
 if m['space']!='head_output_residual_before_post_attention_norm':raise ValueError('Weight-space eigenvectors cannot project residual vectors')
 if not m.get('provenance'):raise ValueError('Gradient provenance required')
 result=[]
 for entry in m['heads']:
  with np.load(p.parent/entry['file'],allow_pickle=False) as z:Q=z[entry.get('key','Q')].astype(np.float64)
  if Q.ndim!=2 or Q.shape[0]!=width or Q.shape[1]>width or not np.isfinite(Q).all():raise ValueError('Invalid basis dimensions')
  if not np.allclose(Q.T@Q,np.eye(Q.shape[1]),atol=1e-5):raise ValueError('Basis must be orthonormal')
  result.append((entry['layer'],entry['head'],Q))
 return result


@torch.no_grad()
def delivered(engine,C,l,write):
 """Inject a local output-projection write at final position; propagate through
 the same frozen block and downstream program, including OLMo post-norm.
 """
 b=engine.blocks[l];proj=b.attention.dense if engine.kind=='gpt_neox' else b.self_attn.o_proj
 x=C['layers'][l]['x'];base=engine.layer(x,l,C)
 def hook(module,args,out):
  y=out.clone();y[-1]+=write;return y
 h=proj.register_forward_hook(hook)
 try:y=engine.layer(x,l,C)-base
 finally:h.remove()
 for j in range(l+1,engine.L):y=engine.linear(y,j,C)
 return y


@torch.no_grad()
def analyze(engine,C,S,g,c,bases,cfg):
 states=engine.component(C,S);rows=[]
 for l,h,q in bases:
  if not (0<=l<engine.L and 0<=h<engine.H):raise ValueError('Basis head out of range')
  b=engine.blocks[l];x=states[l];zero=torch.zeros_like(x);d=C['layers'][l]
  if engine.kind!='olmo2':
   x=norm(x,b.input_layernorm,d['input_layernorm']);zero=norm(zero,b.input_layernorm,d['input_layernorm'])
  v=engine.values(x,l)-engine.values(zero,l)
  proj=b.attention.dense if engine.kind=='gpt_neox' else b.self_attn.o_proj
  write=proj.weight[:,h*engine.dh:(h+1)*engine.dh]@(d['A'][h,-1]@v[h])
  Q=torch.as_tensor(q,device=engine.device,dtype=torch.float32)
  gen=torch.Generator(device='cpu').manual_seed(cfg['seed']+l*10000+h)
  random=torch.linalg.qr(torch.randn(Q.shape,generator=gen),mode='reduced').Q.to(engine.device)
  full=delivered(engine,C,l,write);zero=torch.zeros_like(full)
  def score(y):
   a=engine.read(y,C,[g,c])-engine.read(zero,C,[g,c]);return float(a[0]),float(a[0]-a[1])
  gold,margin=score(full)
  for kind,B in [('pretraining_gradient',Q),('random_rank_matched',random)]:
   part=B@(B.T@write);dpart=delivered(engine,C,l,part);dperp=delivered(engine,C,l,write-part)
   error=float((dpart+dperp-full).norm()/full.norm().clamp_min(1))
   if error>cfg['certificate_tolerance']:raise AssertionError('Projected delivery additivity failed')
   pg,pm=score(dpart);og,om=score(dperp)
   rows.append(dict(layer=l,head=h,basis=kind,rank=Q.shape[1],write_norm=float(write.norm()),
    local_energy_fraction=float(part.square().sum()/write.square().sum().clamp_min(1e-20)),
    delivered_gold=gold,delivered_margin=margin,projected_gold=pg,projected_margin=pm,
    orthogonal_gold=og,orthogonal_margin=om,projected_signed_margin_fraction=pm/margin if abs(margin)>1e-6 else None,certificate=error))
 return rows


def build_bases(manifest,out,rank):
 """Input manifest: actual gradient snapshots (NPZ), keys layer_0, layer_1...
 each a W_o gradient [d_model, num_heads*head_dim]. Accumulates SUM G_h G_h^T,
 not outer products of flattened weights; unweighted snapshot sampling defines
 the estimand. Full d*d Grams per head are processed sequentially on CPU.
 """
 p=Path(manifest);m=json.loads(p.read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if m.get('gradient_origin')!='pretraining' or not m.get('provenance'):raise ValueError('Actual pretraining snapshots and provenance required')
 heads=[]
 for layer in m['layers']:
  for head in range(m['num_heads']):
   gram=None
   for filename in m['snapshots']:
    with np.load(p.parent/filename,allow_pickle=False) as z:G=z[f'layer_{layer}'].astype(np.float64)
    if G.ndim!=2 or G.shape[1]%m['num_heads']:raise ValueError('Invalid output-projection gradient shape')
    dh=G.shape[1]//m['num_heads'];g=G[:,head*dh:(head+1)*dh]
    gram=g@g.T if gram is None else gram+g@g.T
   if gram is None:raise ValueError('No gradient snapshots')
   w,Q=np.linalg.eigh(gram);order=np.argsort(w)[::-1][:rank];Q=Q[:,order]
   name=f'layer{layer}_head{head}.npz';np.savez_compressed(out/name,Q=Q,eigenvalues=w[order])
   heads.append(dict(layer=layer,head=head,file=name,key='Q'))
 dump(out/'manifest.json',dict(model_sha=m['model_sha'],space='head_output_residual_before_post_attention_norm',
  definition='Unweighted sum over supplied pretraining snapshots of G_head @ G_head.T',provenance=m['provenance'],heads=heads))

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--gradients',required=True);a.add_argument('--out',required=True);a.add_argument('--rank',type=int,default=64);args=a.parse_args()
 build_bases(args.gradients,args.out,args.rank)
