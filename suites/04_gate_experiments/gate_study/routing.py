from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np
import torch
from causal_suite.data import digest
from .common import dump,read_meta,meta_path,key,load_arrays
from .factors import role_indices
from .edits import state,endpoints


def redistribute(row,indices,desired):
 """Set a subset to donor values; preserve relative probabilities elsewhere."""
 out=row.clone();ix=torch.as_tensor(indices,device=row.device,dtype=torch.long);mask=torch.ones_like(row,dtype=torch.bool);mask[ix]=False
 mass=desired.sum()
 if mass>1+1e-6 or mass<0:raise ValueError('Invalid transplanted probability mass')
 remaining=row[mask].sum()
 if remaining<1e-12 and 1-mass>1e-6:raise ValueError('No recipient probability mass available for redistribution')
 out[ix]=desired;out[mask]=row[mask]*(1-mass).clamp_min(0)/remaining.clamp_min(1e-12)
 if not torch.isclose(out.sum(),out.new_tensor(1.),atol=1e-5):raise AssertionError('Attention row not on simplex')
 return out


def patch_row(row,donor,S,DS,operation,seed=0):
 roles=role_indices(len(row),S);droles=role_indices(len(donor),DS)
 if operation=='role_mass':
  result=row.clone()
  for ix,jx in zip(roles,droles):
   mass=donor[jx].sum()
   if not ix:
    if mass>1e-6:raise ValueError('Donor role has no recipient counterpart')
    continue
   denom=row[ix].sum()
   if denom<1e-12 and mass>1e-6:raise ValueError('Cannot redistribute a zero-mass role')
   result[ix]=row[ix]*mass/denom.clamp_min(1e-12)
  return result
 if operation=='sink':return redistribute(row,roles[0],donor[droles[0]])
 if len(row)!=len(donor) or list(S)!=list(DS):raise ValueError('Token-level pattern transplant requires exact layout')
 ix=roles[2]+roles[3] if operation in ['template','shuffle_template'] else roles[1]
 desired=donor[ix].clone()
 if operation=='shuffle_template' and len(ix):
  rng=np.random.default_rng(seed);desired=desired[torch.as_tensor(rng.permutation(len(ix)),device=row.device)]
 return redistribute(row,ix,desired)


@contextmanager
def pattern_patch(engine,donor,S,DS,heads,operation,seed=0):
 handles=[]
 for l,b in enumerate(engine.blocks):
  selected=[h for ll,h in heads if ll==l]
  if not selected:continue
  def hook(module,args,kw,out,l=l,b=b,selected=selected):
   hidden=args[0] if args else kw['hidden_states'];x=hidden[0];A=out[1]
   if A is None:raise ValueError('Eager output_attentions=True is required')
   A=A[0].clone()
   for h in selected:A[h,-1]=patch_row(A[h,-1],donor['layers'][l]['A'][h,-1],S,DS,operation,seed)
   if not torch.allclose(A.sum(-1),torch.ones_like(A.sum(-1)),atol=1e-5):raise AssertionError('Attention row normalization failure')
   y=b.self_attn.o_proj((A@engine.values(x,l)).transpose(0,1).reshape(len(x),-1))[None]
   parts=list(out);parts[0]=y;parts[1]=A[None];return tuple(parts)
  handles.append(b.self_attn.register_forward_hook(hook,with_kwargs=True))
 try:yield
 finally:
  for handle in handles:handle.remove()


@torch.no_grad()
def live_with_attentions(engine,meta):
 ids=torch.tensor(meta['input_ids'],device=engine.device)
 return engine.model(ids[None],attention_mask=torch.ones_like(ids[None]),use_cache=False,output_attentions=True).logits[0,-1]


def same_layout(a,b):
 if len(a['input_ids'])!=len(b['input_ids']) or a['source']!=b['source']:return False
 return all(v==w for i,(v,w) in enumerate(zip(a['input_ids'],b['input_ids'])) if i not in a['source'])


def select_heads(ref,k):
 scores=ref['head_margin'];order=np.argsort(scores.ravel())[::-1];positive=order[scores.ravel()[order]>0][:k]
 return [list(np.unravel_index(int(i),scores.shape)) for i in positive]


def run_case(engine,r,calibration,root,index,cfg):
 root=Path(root);path=root/'routing'/f'{key(r)}.json'
 if path.exists():return
 m=read_meta(root,r,0)
 if m.get('status')!='measured' or m['correct']:return
 entry=index.get(r['relation'],{})
 if entry.get('status')!='ready':dump(path,dict(id=m['id'],status='no_independent_head_reference',rows=[]));return
 ref=load_arrays(root/'selectors'/entry['file']);heads=select_heads(ref,cfg['handoff_heads'])
 if not heads:dump(path,dict(id=m['id'],status='no_positive_handoff_heads',rows=[]));return
 donors=[]
 for q in calibration:
  if q['relation']!=r['relation'] or str(q['group_id'])==str(r['group_id']):continue
  if not meta_path(root,q,0).exists():continue
  d=read_meta(root,q,0)
  if d.get('status') in ['measured','calibration_not_success'] and same_layout(m,d):donors.append(d)
 donors.sort(key=lambda d:digest([cfg['seed'],m['id'],d['id']]))
 chosen=[]
 for correct in [True,False]:chosen += [d for d in donors if d['correct']==correct][:cfg['routing_donors']]
 for t in [1,2,3]:
  if meta_path(root,r,t).exists():
   d=read_meta(root,r,t)
   if d.get('status')=='measured':chosen.append(d)
 C=engine.capture(torch.tensor(m['input_ids'],device=engine.device));base=endpoints(C['z'],m);rows=[]
 for donor in chosen:
  D=engine.capture(torch.tensor(donor['input_ids'],device=engine.device));different_template=donor['template']!=0
  operations=['role_mass'] if different_template else ['template','subject','sink','shuffle_template']
  for operation in operations:
   headsets=[('handoff',None,heads)]
   for seed in cfg['control_seeds']:
    rng=np.random.default_rng(int(digest([m['id'],seed,'heads'])[:12],16));pool=[(l,h) for l in range(engine.L) for h in range(engine.H) if [l,h] not in heads]
    selected=[pool[i] for i in rng.choice(len(pool),len(heads),replace=False)];headsets.append(('random_heads',seed,selected))
   for head_kind,seed,chosen_heads in headsets:
    try:
     # Validate every intended row before registering live hooks.
     for l,h in chosen_heads:patch_row(C['layers'][l]['A'][h,-1],D['layers'][l]['A'][h,-1],m['source'],donor['source'],operation,seed or 0)
     with pattern_patch(engine,D,m['source'],donor['source'],chosen_heads,operation,seed or 0):z=live_with_attentions(engine,m)
    except ValueError as ex:
     rows.append(dict(id=m['id'],operation=operation,donor_id=donor['id'],status='unalignable',reason=str(ex)));continue
    ep=endpoints(z,m);sink=float(np.mean([float(C['layers'][l]['A'][h,-1,0]) for l,h in chosen_heads]));dsink=float(np.mean([float(D['layers'][l]['A'][h,-1,0]) for l,h in chosen_heads]))
    rows.append(dict(id=m['id'],group_id=m['group_id'],relation=m['relation'],label=m['label'],status='measured',head_kind=head_kind,seed=seed,operation=operation,heads=chosen_heads,
     donor_id=donor['id'],donor_correct=donor['correct'],donor_template=donor['template'],baseline_sink_share=sink,donor_sink_share=dsink,
     **ep,fixed_margin_gain=ep['fixed_margin']-base['fixed_margin'],top1_margin_gain=ep['top1_margin']-base['top1_margin']))
 restored=live_with_attentions(engine,m)
 if not torch.allclose(restored,C['z'],atol=cfg['tolerance'],rtol=cfg['tolerance']):raise AssertionError('Pattern hook rollback failed')
 dump(path,dict(id=m['id'],status='complete',donors=len(chosen),rows=rows))
