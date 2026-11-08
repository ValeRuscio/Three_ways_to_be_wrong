"""Donor preactivation patches for the distance hypothesis; live recomputation.
No assertion that SiLU has a closed/open threshold.
"""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from causal_suite.data import digest
from .common import key,dump,read_meta
from .selectors import sets_for,select_site
from .factors import mlp_input
from .edits import state,endpoints


@contextmanager
def preactivation_patch(engine,layer,position,neurons,delta):
 def hook(module,args,out):
  y=out.clone();y[:,position,neurons]+=delta;return y
 h=engine.blocks[layer].mlp.gate_proj.register_forward_hook(hook)
 try:yield
 finally:h.remove()


def run_case(engine,r,root,index,cfg,baseline):
 root=Path(root);path=root/'activation_patches'/f'{key(r)}.json'
 if path.exists():return
 m=read_meta(root,r,0)
 if m.get('status')!='measured' or m['correct']:return
 sets=sets_for(r,root,index,cfg);rows=[];C=None;z0,_=baseline.get(m);base=endpoints(z0,m)
 for kind,ref,nref in sets:
  for site in cfg['sites']:
   selected=select_site(ref,site,cfg['top_k'])
   if selected is None:continue
   if C is None:C=engine.capture(torch.tensor(m['input_ids'],device=engine.device))
   l=selected['layer'];pos=m['source'][-1] if site=='subject' else len(m['input_ids'])-1;b=engine.blocks[l]
   x=mlp_input(engine,C,l,C['layers'][l]['x'])[pos];pre=b.mlp.gate_proj(x).detach();ix=np.array(selected['neurons']);desired=torch.tensor(selected['reference_pre'],device=engine.device)
   deficit=desired-pre[ix];keep=deficit>0;ix=ix[keep.cpu().numpy()];deficit=deficit[keep]
   if not len(ix):continue
   distance=(pre[ix]/b.mlp.gate_proj.weight[ix].norm(dim=1).clamp_min(1e-20)).mean()
   arms=[('donor_preactivation',None,ix,deficit)]
   for seed in cfg['control_seeds']:
    rng=np.random.default_rng(int(digest([m['id'],site,seed])[:12],16));pool=np.setdiff1d(np.arange(pre.numel()),ix);ri=rng.choice(pool,len(ix),replace=False)
    arms.append(('random_neurons',seed,ri,deficit))
    rd=torch.tensor(rng.normal(size=len(ix)),device=engine.device,dtype=pre.dtype);rd=rd/rd.norm().clamp_min(1e-20)*deficit.norm();arms.append(('random_direction',seed,ix,rd))
   for alpha in cfg['activation_alphas']:
    for arm,seed,neurons,delta in arms:
     with preactivation_patch(engine,l,pos,neurons,alpha*delta):z,_=state(engine,m)
     ep=endpoints(z,m);rows.append(dict(id=m['id'],group_id=m['group_id'],relation=m['relation'],label=m['label'],set_kind=kind,site=site,layer=l,arm=arm,seed=seed,alpha=alpha,
      reference_subjects=nref,selected_neurons=ix.tolist(),pre_mean=float(pre[ix].mean()),signed_distance_mean=float(distance),deficit_mean=float(deficit.mean()),
      preactivation_delta_norm=float((alpha*delta).norm()),baseline_margin=base['top1_margin'],baseline_rank=m['target_rank'],
      **ep,top1_margin_gain=ep['top1_margin']-base['top1_margin'],fixed_margin_gain=ep['fixed_margin']-base['fixed_margin']))
 restored,_=state(engine,m)
 if not torch.allclose(restored,z0,atol=cfg['tolerance'],rtol=cfg['tolerance']):raise AssertionError('Activation hook rollback failed')
 dump(path,dict(id=m['id'],sets=len(sets),rows=rows))
