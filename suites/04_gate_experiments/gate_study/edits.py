"""Temporary live rank-one weight edits with exact rollback. The down-projection
baseline is NOT ROME: no covariance-preserving ROME objective is implemented.
"""
from pathlib import Path
from contextlib import contextmanager
import json
import numpy as np
import torch
from causal_suite.data import digest
from .common import dump,npz,key,read_meta,cache_path,load_arrays
from .cache import generate
from .factors import mlp_input,native_value_direction
from .selectors import sets_for,select_site


@contextmanager
def weight_patch(engine,layer,kind,neurons,delta):
 module=getattr(engine.blocks[layer].mlp,{'gate':'gate_proj','up':'up_proj','down':'down_proj'}[kind]);w=module.weight
 ix=torch.as_tensor(neurons,device=w.device)
 with torch.no_grad():
  old=(w[:,ix] if kind=='down' else w[ix,:]).clone()
  if delta.shape!=old.shape:raise ValueError('Weight-edit shape mismatch')
  if kind=='down':w[:,ix]=old+delta
  else:w[ix,:]=old+delta
 try:yield float(((w[:,ix] if kind=='down' else w[ix,:])-old).norm())
 finally:
  with torch.no_grad():
   if kind=='down':w[:,ix]=old
   else:w[ix,:]=old
   actual=w[:,ix] if kind=='down' else w[ix,:]
   if not torch.equal(actual,old):raise AssertionError('Weight rollback failed')


def unit(x):return x/x.norm().clamp_min(1e-20)


@torch.no_grad()
def state(engine,meta):
 ids=torch.tensor(meta['input_ids'],device=engine.device)
 output=engine.model(ids[None],attention_mask=torch.ones_like(ids[None]),use_cache=False,output_hidden_states=True)
 return output.logits[0,-1].detach(),output.hidden_states[-1][0,-1].detach()


def endpoints(z,meta):
 g,c=meta['g'],meta['c'];other=z.clone();other[g]=-torch.inf
 return dict(fixed_margin=float(z[g]-z[c]),top1_margin=float(z[g]-other.max()),gold_top1=int(z.argmax())==g,top_token=int(z.argmax()),gold_logit=float(z[g]),competitor_logit=float(z[c]))


class Baselines:
 def __init__(self,engine,root):self.engine=engine;self.root=Path(root)/'native_baselines'
 def get(self,meta):
  p=self.root/f'{digest([meta["id"],meta["template"]])[:24]}.npz'
  if p.exists():
   a=load_arrays(p);return torch.tensor(a['z'],device=self.engine.device),torch.tensor(a['h'],device=self.engine.device)
  z,h=state(self.engine,meta);npz(p,z=z.cpu().numpy(),h=h.cpu().numpy());return z,h


def specificity_records(r,records,root,cfg):
 candidates=[]
 for q in records:
  if q['relation']!=r['relation'] or str(q['group_id'])==str(r['group_id']):continue
  try:m=read_meta(root,q,0)
  except FileNotFoundError:continue
  if m.get('status')=='measured':candidates.append((q,m))
 candidates.sort(key=lambda pair:digest([cfg['seed'],str(r['id']),str(pair[0]['id'])]))
 # Include correct and incorrect controls when available; report strata separately.
 good=[p for p in candidates if p[1]['correct']];bad=[p for p in candidates if not p[1]['correct']]
 n=cfg['specificity_per_case'];selected=good[:(n+1)//2]+bad[:n//2];used={str(q['id']) for q,_ in selected}
 selected += [p for p in candidates if str(p[0]['id']) not in used][:max(0,n-len(selected))]
 return [m for _,m in selected]


@torch.no_grad()
def specificity(engine,controls,baseline):
 rows=[]
 for m in controls:
  z0,_=baseline.get(m);z,_=state(engine,m);p=z0.double().softmax(-1);kl=float((p*(z0.double().log_softmax(-1)-z.double().log_softmax(-1))).sum())
  a=endpoints(z,m);b=endpoints(z0,m)
  rows.append(dict(id=m['id'],baseline_correct=m['correct'],baseline_top1=b['gold_top1'],gold_top1=a['gold_top1'],prediction_changed=a['top_token']!=b['top_token'],top1_margin_gain=a['top1_margin']-b['top1_margin'],kl=kl))
 return rows


def make_arms(engine,C,meta,selected,ref,cfg):
 l=selected['layer'];s=selected['site_index'];pos=meta['source'][-1] if selected['site']=='subject' else len(meta['input_ids'])-1
 b=engine.blocks[l];ix=np.array(selected['neurons']);x=mlp_input(engine,C,l,C['layers'][l]['x'])[pos].detach();pre=b.mlp.gate_proj(x).detach()
 # Neurons lower than the independent/donor reference, continuously weighted.
 deficit=torch.tensor(selected['reference_pre'],device=engine.device)-pre[ix]
 active=deficit>0;ix=ix[active.cpu().numpy()];deficit=deficit[active]
 if not len(ix):return [],dict(reason='No positive preactivation deficit at selected site',selected_neurons=selected['neurons'])
 coeff=unit(deficit);direction=unit(x);rows_norm=float(b.mlp.gate_proj.weight[ix].norm())
 live_r=native_value_direction(engine,C['ids'],l,pos,meta['g'],meta['c']);h=(b.mlp.act_fn(pre)*b.mlp.up_proj(x))[ix].detach()
 value_dir=unit(live_r);value_coeff=unit(h)
 arms=[dict(arm='gate_target',kind='gate',neurons=ix.tolist(),unit_delta=coeff[:,None]*direction[None],seed=None),
       dict(arm='gate_reverse',kind='gate',neurons=ix.tolist(),unit_delta=-coeff[:,None]*direction[None],seed=None)]
 if live_r.norm()>1e-12 and h.norm()>1e-12:
  arms.append(dict(arm='down_target',kind='down',neurons=ix.tolist(),unit_delta=value_dir[:,None]*value_coeff[None],seed=None))
 # Up-projection baseline distinguishes changing the gate from changing its read.
 with torch.no_grad():
  up_coeff=unit(b.mlp.act_fn(pre)[ix]*(live_r@b.mlp.down_proj.weight[:,ix]))
 if up_coeff.norm()>1e-12:arms.append(dict(arm='up_target',kind='up',neurons=ix.tolist(),unit_delta=up_coeff[:,None]*direction[None],seed=None))
 available=np.setdiff1d(np.arange(pre.numel()),ix)
 for seed in cfg['control_seeds']:
  rng=np.random.default_rng(int(digest([cfg['seed'],meta['id'],l,selected['site'],seed])[:12],16));random_ix=rng.choice(available,size=len(ix),replace=False)
  rx=unit(torch.tensor(rng.normal(size=x.numel()),device=engine.device,dtype=x.dtype));rv=unit(torch.tensor(rng.normal(size=live_r.numel()),device=engine.device,dtype=x.dtype))
  arms += [dict(arm='gate_random_neurons',kind='gate',neurons=random_ix.tolist(),unit_delta=coeff[:,None]*direction[None],seed=seed),
           dict(arm='gate_random_direction',kind='gate',neurons=ix.tolist(),unit_delta=coeff[:,None]*rx[None],seed=seed)]
  if live_r.norm()>1e-12 and h.norm()>1e-12:
   arms += [dict(arm='down_random_neurons',kind='down',neurons=random_ix.tolist(),unit_delta=value_dir[:,None]*value_coeff[None],seed=seed),
            dict(arm='down_random_direction',kind='down',neurons=ix.tolist(),unit_delta=rv[:,None]*value_coeff[None],seed=seed)]
 return arms,dict(layer=l,position=pos,active_neurons=ix.tolist(),selected_neurons=selected['neurons'],reference_pre=selected['reference_pre'],
  available_arms=[a['arm'] for a in arms],live_value_gradient_norm=float(live_r.norm()),selected_native_activation_norm=float(h.norm()),
  deficit_mean=float(deficit.mean()),gate_row_block_norm=rows_norm,input_norm=float(x.norm()),signed_distance_mean=float((pre[ix]/b.mlp.gate_proj.weight[ix].norm(dim=1).clamp_min(1e-20)).mean()),
  pre_mean=float(pre[ix].mean()),same_norm_definition='Frobenius norm rho * original selected W_gate row-block norm; identical absolute delta norm for every arm')


def run_case(engine,tok,r,records,root,index,cfg,baseline):
 meta=read_meta(root,r,0)
 if meta.get('status')!='measured' or meta['correct']:return
 root=Path(root);folder=root/'weight_edits'/key(r);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'coverage.json').exists():return
 sets=sets_for(r,root,index,cfg);controls=specificity_records(r,records,root,cfg)
 # Prime unedited controls BEFORE any temporary patch.
 for m in controls:baseline.get(m)
 z0,h0=baseline.get(meta);base=endpoints(z0,meta)
 C=None;coverage=[]
 for kind,ref,nref in sets:
  for site in cfg['sites']:
   selected=select_site(ref,site,cfg['top_k'])
   if selected is None:coverage.append(dict(set_kind=kind,site=site,status='no_positive_reference_neurons'));continue
   if C is None:C=engine.capture(torch.tensor(meta['input_ids'],device=engine.device),tol=cfg['tolerance'])
   arms,info=make_arms(engine,C,meta,selected,ref,cfg)
   coverage.append(dict(set_kind=kind,site=site,status='ready' if arms else 'no_positive_deficit',reference_subjects=nref,**info))
   for rho in cfg['weight_relative_norms']:
    budget=rho*info.get('gate_row_block_norm',0)
    for arm in arms:
     filename=f'{kind}_{site}_{rho}_{arm["arm"]}_{arm["seed"]}.json';path=folder/filename
     if path.exists():continue
     delta=budget*arm['unit_delta'];delta_norm=float(delta.norm())
     if abs(delta_norm-budget)>1e-5*max(1,budget):raise AssertionError('Norm matching failed')
     l=selected['layer'];x=mlp_input(engine,C,l,C['layers'][l]['x'])[info['position']].detach()
     pre_before=engine.blocks[l].mlp.gate_proj(x)[info['active_neurons']].detach()
     with weight_patch(engine,l,arm['kind'],arm['neurons'],delta) as actual_norm:
      if abs(actual_norm-budget)>1e-5*max(1,budget):raise AssertionError('Applied weight norm differs from planned budget')
      z,h=state(engine,meta);metrics=endpoints(z,meta)
      pre_after=engine.blocks[l].mlp.gate_proj(x)[info['active_neurons']].detach()
      generated=None;answer_correct=None
      if cfg['generate_on_gold_top1'] and metrics['gold_top1']:generated,answer_correct=generate(engine,tok,C['ids'],meta['aliases'],cfg)
      specific=specificity(engine,controls,baseline)
     row=dict(id=meta['id'],group_id=meta['group_id'],relation=meta['relation'],label=meta['label'],set_kind=kind,reference_subjects=nref,site=site,layer=l,
      arm=arm['arm'],seed=arm['seed'],rho=rho,delta_norm=actual_norm,planned_delta_norm=delta_norm,neurons=arm['neurons'],
      baseline=base,**metrics,fixed_margin_gain=metrics['fixed_margin']-base['fixed_margin'],top1_margin_gain=metrics['top1_margin']-base['top1_margin'],
      preactivation_gain=float((pre_after-pre_before).mean()),final_normalized_relative_change=float((h-h0).norm()/h0.norm().clamp_min(1e-12)),
      generated=generated,full_answer_correct=answer_correct,generation_condition='Only generated when canonical gold is top-1; null is unmeasured, not a wrong answer',specificity=specific,**{k:v for k,v in info.items() if k!='layer'})
     dump(path,row)
   # Exact rollback plus fresh native output check after all edits for this site.
   restored,_=state(engine,meta)
   if not torch.allclose(restored,z0,atol=cfg['tolerance'],rtol=cfg['tolerance']):raise AssertionError('Native output did not restore after weight edits')
 dump(folder/'coverage.json',dict(id=meta['id'],sets=len(sets),specificity_controls=len(controls),rows=coverage))
