from pathlib import Path
from contextlib import nullcontext
import torch
from .common import *
from .interventions import gates,attention,frozen,align_rows,edge_gradient,interpolate_edges
from .fast_curves import FastCurves


def ordered_indices(scores,order,seed):
 if order=='contribution':return np.argsort(-np.abs(scores).reshape(-1),kind='stable')
 return np.random.default_rng(seed).permutation(scores.size)


def run(e,root,cfg):
 ps=pairs(root,reverse=True);dump(Path(root)/'curve_pairs.json',ps)
 for pi,pair in enumerate(ps):
  folder=Path(root)/'curves'/pair['pair_id'];folder.mkdir(parents=True,exist_ok=True)
  if (folder/'DONE.json').exists():continue
  m=pair['recipient'];d=pair['donor'];a=load_arrays(cache_path(root,m,m['template']));b=load_arrays(cache_path(root,d,d['template']))
  C=e.capture(ids(e,m),cfg['tolerance']);base=endpoint(C['z'],m);G=a['gate'][:,1];D=b['gate'][:,1]
  fast=FastCurves(e,C,m,cfg['tolerance']);enabled=fast.check(folder/'fast_validation.json')
  print(f'Curve {pi+1}/{len(ps)}: cached final-token execution={enabled}',flush=True)
  score=G*a['up_subject'][:,1]*a['down_margin'][:,1]
  scopes={f'layer_{l}':[l] for l in range(e.L)};scopes['all']=list(range(e.L))
  for scope,ls in scopes.items():
   for order in cfg['curve_orders']:
    for seed in (cfg['seeds'] if order=='random' else [None]):
     inds=ordered_indices(score[ls],order,seed);ks=sorted(set([k for k in cfg['curve_k'] if k<len(inds)]+[len(inds)]))
     for mode in cfg['curve_modes']:
      path=folder/(stamp('gates',scope,order,seed,mode)+'.json')
      if path.exists():continue
      rows=[]
      for k in ks:
       choices=inds[:k];changes={}
       for i,l in enumerate(ls):
        ix=choices[(choices//G.shape[1])==i]%G.shape[1];changes[l]=(ix,D[l,ix])
       z=fast.evaluate(mode,gchanges=changes)
       ep=endpoint(z,m);rows.append(dict(k=k,**ep,margin_change=ep['margin']-base['margin']))
      dump(path,dict(pair_id=pair['pair_id'],id=m['id'],group_id=m['group_id'],relation=m['relation'],direction=pair['direction'],operation='gate_donor',scope=scope,order=order,seed=seed,mode=mode,baseline=base,rows=rows))
   for operation in ['binarize_recipient','donor_sign_recipient_magnitude','donor_open_mask_recipient_magnitude','recipient_sign_donor_magnitude']:
    for mode in cfg['curve_modes']:
     path=folder/(stamp(operation,scope,mode)+'.json')
     if path.exists():continue
     changes={}
     for l in ls:
      if operation=='binarize_recipient':
       positive=G[l]>0;v=positive*(float(G[l,positive].mean()) if positive.any() else 0.)
      elif operation=='donor_sign_recipient_magnitude':v=np.sign(D[l])*np.abs(G[l])
      elif operation=='donor_open_mask_recipient_magnitude':v=(D[l]>0)*np.abs(G[l])
      else:v=np.sign(G[l])*np.abs(D[l])
      changes[l]=(np.arange(len(v)),v)
     z=fast.evaluate(mode,gchanges=changes)
     ep=endpoint(z,m)
     dump(path,dict(pair_id=pair['pair_id'],id=m['id'],group_id=m['group_id'],relation=m['relation'],direction=pair['direction'],operation=operation,scope=scope,mode=mode,baseline=base,endpoint=ep,margin_change=ep['margin']-base['margin']))
  # Cross-template edges require explicit coordinate transport, never silent padding.
  try:AD=align_rows(b['attention_rows'],a['attention_rows'],d['source'],m['source'])
  except ValueError as ex:dump(folder/'edge_coverage.json',dict(status='excluded',reason=str(ex)))
  else:
   # Four full-program hybrids with exact frozen origin ledgers. These freeze
   # recipient norms and keep recipient token inputs; donor A is role-aligned.
   for use_gate in [False,True]:
    for use_attention in [False,True]:
     hp=folder/(stamp('hybrid',use_gate,use_attention)+'.json')
     if hp.exists():continue
     H=dict(C,layers=[dict(x) for x in C['layers']])
     for l in range(e.L):
      if use_gate:
       gate=H['layers'][l]['gate'].clone();gate[-1]=torch.tensor(D[l],device=e.device);H['layers'][l]['gate']=gate
      if use_attention:
       att=H['layers'][l]['A'].clone();att[:,-1]=torch.tensor(AD[l],device=e.device);H['layers'][l]['A']=att
     H['z']=frozen(e,H)
     ledger=e.ledger(H,m['source'],[m['g'],m['c']],[],cfg['tolerance'])
     origins={k:ledger[str(m['g'])]['origins'][k]-ledger[str(m['c'])]['origins'][k] for k in ledger[str(m['g'])]['origins']}
     dump(hp,dict(pair_id=pair['pair_id'],id=m['id'],group_id=m['group_id'],relation=m['relation'],direction=pair['direction'],operation='full_hybrid',mode='frozen',scope='all',donor_gate=use_gate,donor_attention=use_attention,baseline=base,endpoint=endpoint(H['z'],m),origin_margin=origins,margin_change=endpoint(H['z'],m)['margin']-base['margin']))
   A=a['attention_rows'];sens=edge_gradient(e,C,m['g'],m['c']);escore=sens*(AD-A)
   for scope,ls in scopes.items():
    for order in cfg['curve_orders']:
     for seed in (cfg['seeds'] if order=='random' else [None]):
      inds=ordered_indices(escore[ls],order,seed);ks=sorted(set([k for k in cfg['curve_k'] if k<len(inds)]+[len(inds)]))
      for mode in cfg['curve_modes']:
       path=folder/(stamp('edges',scope,order,seed,mode)+'.json')
       if path.exists():continue
       rows=[]
       for k in ks:
        new=interpolate_edges(A[ls],AD[ls],inds[:k]);changes={l:new[i] for i,l in enumerate(ls)}
        z=fast.evaluate(mode,rows=changes)
        ep=endpoint(z,m);rows.append(dict(k=k,**ep,margin_change=ep['margin']-base['margin']))
       dump(path,dict(pair_id=pair['pair_id'],id=m['id'],group_id=m['group_id'],relation=m['relation'],direction=pair['direction'],operation='role_aligned_edge_donor',scope=scope,order=order,seed=seed,mode=mode,baseline=base,alignment='monotone mass transport within semantic roles',rows=rows))
  dump(folder/'DONE.json',dict(status='complete'))
  print(f'Gate/edge curves {pi+1}/{len(ps)} saved',flush=True)
