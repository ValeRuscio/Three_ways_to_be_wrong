"""Optional actual-pretraining gate-gradient geometry. Never synthesizes missing
training gradients from a released checkpoint or a fresh PopQA backward pass.
"""
from pathlib import Path
import json,argparse
import numpy as np
from .common import dump,npz,table


def build(manifest,out,rank=64):
 p=Path(manifest);m=json.loads(p.read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if m.get('gradient_origin')!='pretraining' or not m.get('provenance'):raise ValueError('Actual pretraining gradients and provenance are required')
 basis_steps={int(x['step']) for x in m['basis_snapshots']};probe_steps={int(x['step']) for x in m['probe_snapshots']}
 if not basis_steps or not probe_steps or basis_steps & probe_steps:raise ValueError('Basis and probe training steps must be nonempty and disjoint')
 entries=[]
 for layer in m['layers']:
  gram=None
  for snapshot in m['basis_snapshots']:
   with np.load(p.parent/snapshot['file'],allow_pickle=False) as z:G=z[f'gate_layer_{layer}'].astype(np.float64)
   if G.ndim!=2 or not np.isfinite(G).all():raise ValueError('Gate gradient must be finite [neurons,d_model]')
   gram=G.T@G if gram is None else gram+G.T@G
  if rank<1 or rank>len(gram):raise ValueError('Invalid basis rank')
  w,Q=np.linalg.eigh(gram);ii=np.argsort(w)[::-1][:rank];filename=f'gate_input_layer_{layer}.npz';npz(out/filename,Q=Q[:,ii],eigenvalues=w[ii]);entries.append(dict(layer=layer,file=filename))
 dump(out/'manifest.json',dict(repo=m['repo'],gradient_origin='pretraining',provenance=m['provenance'],space='gate_row_gradient_input_coordinates',basis_steps=sorted(basis_steps),
  bases=entries,probe_snapshots=[dict(s,file=str((p.parent/s['file']).resolve())) for s in m['probe_snapshots']]))


def analyze(manifest,training_root,cfg):
 root=Path(training_root);out=root/'reports'
 if not manifest:dump(out/'gradient_status.json',dict(status='not_run',reason='Actual pretraining gate-gradient snapshots and held-out-step basis were not supplied'));return
 p=Path(manifest);m=json.loads(p.read_text())
 if m.get('gradient_origin')!='pretraining' or m.get('space')!='gate_row_gradient_input_coordinates' or not m.get('provenance'):raise ValueError('Invalid pretraining gradient manifest')
 if m['repo']!=cfg['training_repo']:raise ValueError('Gradient training run/model repository mismatch')
 if set(m['basis_steps']) & {s['step'] for s in m['probe_snapshots']}:raise ValueError('Probe gradients were used to estimate their own subspace')
 fixed=json.loads((root/'fixed_neurons.json').read_text());chosen={}
 for record in fixed.values():
  for s in record['sites']:chosen.setdefault((record['kind'],s['layer']),set()).update(s['neurons'])
 rows=[]
 for entry in m['bases']:
  l=entry['layer']
  with np.load(p.parent/entry['file'],allow_pickle=False) as z:Q=z['Q'].astype(float)
  if not np.allclose(Q.T@Q,np.eye(Q.shape[1]),atol=1e-5):raise ValueError('Non-orthonormal gradient basis')
  for snap in m['probe_snapshots']:
   file=Path(snap['file']);file=file if file.is_absolute() else p.parent/file
   with np.load(file,allow_pickle=False) as z:G=z[f'gate_layer_{l}'].astype(float)
   if G.shape[1]!=Q.shape[0]:raise ValueError('Gradient/basis coordinate mismatch')
   for (kind,ll),indices in chosen.items():
    if ll!=l:continue
    selected=np.array(sorted(indices),dtype=int)
    if len(selected) and selected.max()>=len(G):raise ValueError('Neuron IDs outside supplied gradient shape')
    groups=[('selected',None,selected)];pool=np.setdiff1d(np.arange(len(G)),selected)
    for seed in cfg['control_seeds']:
     rng=np.random.default_rng(seed+l*1000+snap['step']);groups.append(('random_neurons',seed,rng.choice(pool,min(len(pool),len(selected)),replace=False)))
    for group,seed,ix in groups:
     norm=np.square(G[ix]).sum(-1);energy=np.square(G[ix]@Q).sum(-1)
     for neuron,n,e in zip(ix,norm,energy):rows.append(dict(step=snap['step'],layer=l,neuron=int(neuron),set_kind=kind,group=group,seed=seed,rank=Q.shape[1],gradient_norm=float(np.sqrt(n)),subspace_energy_fraction=float(e/n) if n>1e-20 else None))
 table(rows,out/'training_gradient_alignment.csv');dump(out/'gradient_status.json',dict(status='complete',provenance=m['provenance'],unit='Unique neuron per layer/step/set kind; repeated fact memberships are not independent samples'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=64);a=p.parse_args();build(a.manifest,a.out,a.rank)
