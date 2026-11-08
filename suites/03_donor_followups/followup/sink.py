"""Exact full-vocabulary first-position/subject/template origin vectors.
Streaming statistics over all vocabulary tokens, not a handpicked token subset.
A first-position origin is not itself an attention-sink measurement; both logged.
"""
from pathlib import Path
import json
import numpy as np
import torch
from causal_suite.engine import norm
from .common import dump,csv


@torch.no_grad()
def normalized_origin_vectors(engine,C,S):
 T=len(C['ids']);groups={'subject':S,'first_token':[] if 0 in S else [0],'template':[i for i in range(T) if i not in S and i!=0]}
 vectors={}
 for name,positions in groups.items():
  component=engine.component(C,positions)[-1][-1]
  vectors[name]=norm(component,engine.final,C['inv'][-1])-norm(torch.zeros_like(component),engine.final,C['inv'][-1])
 zero=torch.zeros_like(C['layers'][0]['x'])
 for l in range(engine.L):zero=engine.layer(zero,l,C)
 bias=norm(zero[-1],engine.final,C['inv'][-1])
 vectors['bias']=bias
 return vectors


@torch.no_grad()
def vector_snapshot(engine,C,S,path,metadata,cfg):
 vectors=normalized_origin_vectors(engine,C,S);store={};size=engine.head.weight.shape[0]
 for name,vec in vectors.items():
  pieces=[]
  for start in range(0,size,cfg['sink_vocab_block']):
   piece=engine.head.weight[start:start+cfg['sink_vocab_block']]@vec
   if name=='bias' and engine.head.bias is not None:piece+=engine.head.bias[start:start+cfg['sink_vocab_block']]
   pieces.append(piece.cpu().numpy())
  store[name]=np.concatenate(pieces).astype(np.float32)
 total=sum(store.values());z=C['z'].cpu().numpy();err=float(np.linalg.norm(total-z)/max(1,np.linalg.norm(z)))
 if err>cfg['certificate_tolerance']:raise AssertionError('Full-vocabulary origin certificate failed')
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with open(tmp,'wb') as f:np.savez_compressed(f,**store)
 tmp.replace(path)
 attention_to_first=[float(d['A'][:,-1,0].mean()) for d in C['layers']]
 meta=dict(metadata,certificate=err,first_token_id=int(C['ids'][0]),subject_includes_first=0 in S,
           mean_attention_to_first=float(np.mean(attention_to_first)),attention_to_first_by_layer=attention_to_first)
 dump(path.with_suffix('.json'),meta)


def frequency_vector(path,vocab_size,model_sha,tokenizer_hash):
 if path is None:return None,{'status':'not_run','reason':'No independent frequency counts/direction supplied'}
 path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
 if meta['tokenizer_hash']!=tokenizer_hash:raise ValueError('Frequency tokenizer fingerprint mismatch')
 with np.load(path,allow_pickle=False) as d:
  if 'counts' in d:
   counts=d['counts'].astype(float)
   if (counts<0).any():raise ValueError('Negative frequency count')
   direction=np.log(counts+.5)
  elif 'log_frequency' in d:direction=d['log_frequency'].astype(float)
  else:raise ValueError('Frequency NPZ requires counts or log_frequency in full-vocabulary token-ID order')
 if direction.shape!=(vocab_size,) or not np.isfinite(direction).all():raise ValueError('Bad vocabulary frequency shape/values')
 direction-=direction.mean();n=np.linalg.norm(direction)
 if n==0:raise ValueError('Constant frequency direction')
 return direction/n,dict(status='loaded',**meta)


def aggregate_snapshots(directory,out,frequency=None):
 """Within relation/template centered logit-vector variation; token-independent
 denominator total second moment. Center across vocab to remove arbitrary logit
 offsets. Equal-position input embedding does not imply equal downstream map.
 """
 paths=sorted(Path(directory).glob('*.npz'));groups={};rows=[];summaries=[]
 for path in paths:
  meta=json.loads(path.with_suffix('.json').read_text())
  if meta.get('subject_includes_first'):continue
  # Raw primary template is fixed by relation; explicit template hash guards imports.
  key=(meta['relation'],meta['template_key']);groups.setdefault(key,[]).append((path,meta))
 for key,entries in groups.items():
  stats={};references={}
  for path,meta in entries:
   with np.load(path,allow_pickle=False) as data:
    for origin in ['first_token','subject','template']:
     x=data[origin].astype(np.float64);x-=x.mean();n=float(np.linalg.norm(x))
     if origin not in stats:stats[origin]=dict(n=0,mean=np.zeros_like(x),M2=np.zeros_like(x),normsq=0.)
     s=stats[origin];s['n']+=1;delta=x-s['mean'];s['mean']+=delta/s['n'];s['M2']+=delta*(x-s['mean']);s['normsq']+=n*n
     alignment=float(x@frequency/max(n,1e-12)) if frequency is not None and n>1e-12 else None
     rows.append(dict(**meta,origin=origin,centered_norm=n,frequency_cosine=alignment))
  for origin,s in stats.items():
   n=s['n'];m=s['mean'];within=s['M2'].sum()/n;second=s['normsq']/n
   references[origin]=dict(relation=key[0],template_key=key[1],origin=origin,n=n,mean_vector_norm=float(np.linalg.norm(m)),within_vector_rms=float(np.sqrt(max(0,within))),
    variation_fraction=float(within/second) if second>1e-20 and n>1 else None,frequency_cosine_of_mean=float(m@frequency/max(np.linalg.norm(m),1e-12)) if frequency is not None and np.linalg.norm(m)>1e-12 else None,
    interpretation='Within-template centered-vocabulary variance / second moment; lower indicates more invariant. Not a causal bias proof.')
  # Append through local list below, avoiding full example-by-vocabulary matrices.
  for v in references.values():summaries.append(v)
 csv(rows,Path(out)/'sink_per_example.csv');csv(summaries,Path(out)/'sink_template_variation.csv')
 return rows,summaries
