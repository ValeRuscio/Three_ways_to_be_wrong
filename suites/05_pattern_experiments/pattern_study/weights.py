from pathlib import Path
import torch
from .common import *
from . import lowrank


def fit_bank(pair,calpairs,root,cfg):
 selected=[];seen={pair['recipient']['group_id']}
 for p in sorted(calpairs,key=lambda p:stamp(pair['pair_id'],p['pair_id'])):
  m=p['recipient']
  if m['relation']!=pair['recipient']['relation'] or m['group_id'] in seen:continue
  seen.add(m['group_id']);selected.append(p)
  if len(selected)>=cfg['fit_calibration_pairs']:break
 bank=[]
 for p in [pair]+selected:
  f=load_arrays(cache_path(root,p['recipient'],p['recipient']['template']));s=load_arrays(cache_path(root,p['donor'],p['donor']['template']))
  bank.append((f['x'][:,1],s['pre'][:,1]-f['pre'][:,1]))
 return bank,selected


def run(e,tok,root,cfg):
 ps=pairs(root);cp=pairs(root,'calibration');dump(Path(root)/'weight_pairs.json',ps)
 for pi,pair in enumerate(ps):
  folder=Path(root)/'weights'/pair['pair_id'];folder.mkdir(parents=True,exist_ok=True)
  if (folder/'DONE.json').exists():continue
  m=pair['recipient'];s=pair['donor'];bank,selected=fit_bank(pair,cp,root,cfg)
  base=endpoint(logits(e,m),m);controls=baseline_controls(e,control_metas(root,m,cfg['specificity_n']))
  full_layers=list(range(e.L));grad={kind:lowrank.gradients(e,m,kind,full_layers) for kind in ['gate','up','down']}
  for scope in cfg['scopes']:
   ls=layers(e,scope,cfg)
   # One common absolute budget based on the ORIGINAL gate-weight blocks.
   scale=sum(float(e.blocks[l].mlp.gate_proj.weight.norm())**2 for l in ls)**.5
   for rank in cfg['ranks']:
    arms=[];pattern_ranks={}
    for fit_name,entries in [('single_pair',bank[:1]),('pooled_with_target',bank),('calibration_only',bank[1:])]:
     if not entries or (fit_name=='single_pair' and rank!=1):continue
     bs={}
     for l in ls:
      X=torch.tensor(np.stack([v[0][l] for v in entries],axis=1));D=torch.tensor(np.stack([v[1][l] for v in entries],axis=1))
      bs[l]=lowrank.fit(X,D,rank,cfg['ridge'])
     arms.append((fit_name,'gate',None,bs))
     if fit_name=='pooled_with_target':pattern_ranks={l:uv[0].shape[1] for l,uv in bs.items()}
    for kind in ['gate','down','up']:
     bs={l:lowrank.compress(*grad[kind][l],rank) for l in ls}
     arms.append(('gradient_'+kind,kind,None,bs))
    for seed in cfg['seeds']:
     gen=torch.Generator().manual_seed(int(stamp(pair['pair_id'],scope,rank,seed),16)%(2**63-1))
     bs={l:lowrank.random_delta(*e.blocks[l].mlp.gate_proj.weight.shape,pattern_ranks[l],gen) for l in ls}
     arms.append(('random_gate','gate',seed,bs))
    for name,kind,seed,bs in arms:
     for rho in cfg['doses']:
      file=folder/(stamp(scope,rank,name,seed,rho)+'.json')
      if file.exists():continue
      budget=rho*scale;edit=lowrank.normalize(bs,budget)
      common=dict(pair_id=pair['pair_id'],id=m['id'],group_id=m['group_id'],relation=m['relation'],recipient_template=m['template'],donor_template=s['template'],scope=scope,layers=ls,
       rank_requested=rank,arm=name,kind=kind,seed=seed,rho=rho,budget=budget,calibration_pairs=len(selected),baseline=base)
      if edit is None:dump(file,dict(**common,status='zero_direction'));continue
      with lowrank.apply(e,kind,edit):
       z=logits(e,m);a=endpoint(z,m);a.update(generation(e,tok,m,cfg));checks=specificity(e,controls)
       # Recapture selected gate-projection outputs on failing prompt after edit.
       actual={};handles=[]
       try:
        for l in ls:
         def hook(mod,args,out,l=l):actual[l]=out[0,len(m['input_ids'])-1].detach().cpu()
         handles.append(e.blocks[l].mlp.gate_proj.register_forward_hook(hook))
        logits(e,m)
       finally:
        for h in handles:h.remove()
      donor=load_arrays(cache_path(root,s,s['template']))['pre'][:,1]
      target_error=sum(float((actual[l]-torch.tensor(donor[l])).square().sum()) for l in ls)**.5
      dump(file,dict(**common,status='measured',endpoint=a,margin_gain=a['margin']-base['margin'],specificity=checks,
       actual_delta_norm=sum(lowrank.frob(*v)**2 for v in edit.values())**.5,
       effective_ranks={str(l):v[0].shape[1] for l,v in edit.items()},live_pattern_error=target_error))
      print(f'Weight {pi+1}/{len(ps)} {scope} r={rank} {name} rho={rho} saved',flush=True)
  dump(folder/'DONE.json',dict(status='complete',calibration_ids=[p['recipient']['id'] for p in selected]))
