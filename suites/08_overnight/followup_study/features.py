"""One capture, compact exact sufficient statistics; no full gate cache per prompt."""
from pathlib import Path
import json
import numpy as np
import torch
from channel_study.core import mlp_input
from channel_study.io import dump


@torch.no_grad()
def feature_vectors(e,C,S):
 # Gates at the subject-last and answer token. Attention compares semantic masses.
 sites=[S[-1],len(C['ids'])-1];gates=torch.stack([d['gate'][sites] for d in C['layers']]).cpu().numpy()
 T=len(C['ids']);sets=[list(S),[] if 0 in S else [0],[i for i in range(T) if i not in S and i!=0 and i!=T-1],[] if T-1 in S or T-1==0 else [T-1]]
 # Subject precedence; template bucket excludes answer position only when distinct.
 masses=torch.stack([torch.stack([d['A'][:,-1,ix].sum(-1) for ix in sets],-1) for d in C['layers']]).cpu().numpy()
 if not np.allclose(masses.sum(-1),1,atol=1e-5):raise AssertionError('Role mass partition')
 return dict(gate=gates,attention=masses)


@torch.no_grad()
def origin_vectors(e,C,S,g,c,tol):
 x0=C['layers'][0]['x'];groups={'subject':list(S),'first':[] if 0 in S else [0],
  'template':[i for i in range(len(x0)) if i not in S and i!=0]}
 states={}
 for k,ix in groups.items():x=torch.zeros_like(x0);x[ix]=x0[ix];states[k]=x
 bias=torch.zeros_like(x0);pre=[];cert=[]
 for l,b in enumerate(e.blocks):
  zero=torch.zeros_like(x0);mi0=mlp_input(e,C,l,zero)
  inputs={k:mlp_input(e,C,l,x)-mi0 for k,x in states.items()}
  P={k:b.mlp.gate_proj.weight@x[-1] for k,x in inputs.items()}
  pb=b.mlp.gate_proj(mlp_input(e,C,l,bias))[-1]
  actual=b.mlp.gate_proj(mlp_input(e,C,l,C['layers'][l]['x']))[-1]
  err=float((sum(P.values())+pb-actual).norm()/actual.norm().clamp_min(1))
  if err>tol:raise AssertionError('Gate origin reconstruction failed')
  cert.append(err);pre.append(torch.stack([P['subject'],P['template'],P['first']]).cpu().numpy())
  states={k:e.linear(x,l,C) for k,x in states.items()};bias=e.layer(bias,l,C)
 zero=torch.zeros_like(x0);read0=e.read(zero,C)
 logits={k:e.read(x,C)-read0 for k,x in states.items()};logits['bias']=e.read(bias,C)
 recovered=sum(logits.values());err=float((recovered-C['z']).norm()/C['z'].norm().clamp_min(1))
 if err>tol:raise AssertionError('Full vocabulary origin certificate failed')
 centered={k:v-v.mean() for k,v in logits.items()}
 gold={k:float(v[g]) for k,v in centered.items()};competitor={k:float(v[c]) for k,v in centered.items()}
 margin={k:float(v[g]-v[c]) for k,v in logits.items()}
 gd=sum(abs(v) for v in gold.values());cd=sum(abs(v) for v in competitor.values())
 return dict(centered_gold=gold,centered_competitor=competitor,margin_origins=margin,
  gold_source_share=gold['subject']/gd if gd>1e-12 else None,
  competitor_source_share=competitor['subject']/cd if cd>1e-12 else None,
  certificate=err,gate_certificate=max(cert)),np.stack(pre)


def atomic_npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 tmp.replace(path)


class Crossed:
 """Trace ANOVA identities on balanced subject x template vector observations."""
 def __init__(self,path):
  self.path=Path(path);self.done=[];self.data={}
  if self.path.exists():
   with np.load(self.path,allow_pickle=False) as z:
    self.done=json.loads(str(z['done']));self.data={k:z[k] for k in z.files if k!='done'}
 def add(self,subject,features):
  if subject in self.done:return
  # features: K lists of layer/site/vector arrays. Compute each coordinate group trace.
  for name,values in features.items():
   X=np.asarray(values,dtype=np.float64) # K,L,site,coordinate
   sums=X.sum(0);sq=(X*X).sum((0,-1));subsq=(sums*sums).sum(-1)/len(X)
   if name+'_sum' not in self.data:
    self.data[name+'_sum']=np.zeros_like(X)
    self.data[name+'_sq']=np.zeros_like(sq);self.data[name+'_subsq']=np.zeros_like(subsq)
   self.data[name+'_sum']+=X;self.data[name+'_sq']+=sq;self.data[name+'_subsq']+=subsq
  self.done.append(subject);atomic_npz(self.path,done=json.dumps(self.done),**self.data)
 def rows(self):
  n=len(self.done);rows=[]
  if n<2:return rows
  for name in ['gate','attention']:
   if name+'_sum' not in self.data:continue
   sums=self.data[name+'_sum'];K=len(sums);grand=(sums.sum(0)**2).sum(-1)/(n*K)
   total=self.data[name+'_sq']-grand;subject=self.data[name+'_subsq']-grand
   template=(sums*sums).sum((0,-1))/n-grand;interaction=total-subject-template
   scale=np.maximum(1.,np.abs(total))
   if np.any(np.minimum(np.minimum(subject,template),interaction)<-1e-8*scale):raise AssertionError('Negative ANOVA energy beyond rounding tolerance')
   for l in range(total.shape[0]):
    for site in range(total.shape[1]):
     row=dict(measurement=name,layer=l,site=site,subjects=n,templates=K,total_ss=float(total[l,site]))
     for key,value in [('subject',subject),('template',template),('interaction',interaction)]:
      row[key+'_ss']=float(max(0,value[l,site]));row[key+'_fraction']=float(max(0,value[l,site])/total[l,site]) if total[l,site]>1e-12 else None
     rows.append(row)
  return rows


class Moments:
 """Per-template vector means, trace variance, exact mean pairwise cosine."""
 def __init__(self,path):
  self.path=Path(path);self.done=[];self.data={}
  if self.path.exists():
   with np.load(self.path,allow_pickle=False) as z:self.done=json.loads(str(z['done']));self.data={k:z[k] for k in z.files if k!='done'}
 def add(self,subject,X):
  if subject in self.done:return
  X=np.asarray(X,dtype=np.float64) # K,L,3,neurons
  norms=np.linalg.norm(X,axis=-1);valid=norms>1e-12;unit=X/np.maximum(norms[...,None],1e-12)
  for name,value in dict(sum=X,sq=norms**2,unit=unit,valid=valid.astype(float)).items():
   if name not in self.data:self.data[name]=np.zeros_like(value)
   self.data[name]+=value
  self.done.append(subject);atomic_npz(self.path,done=json.dumps(self.done),**self.data)
 def rows(self):
  n=len(self.done);rows=[]
  if not n:return rows
  mean_energy=(self.data['sum']/n)**2;mean_energy=mean_energy.sum(-1)
  total=self.data['sq']/n;variation=np.maximum(0,total-mean_energy);v=self.data['valid']
  pair=((self.data['unit']**2).sum(-1)-v)/np.maximum(1,v*(v-1))
  for t in range(total.shape[0]):
   for l in range(total.shape[1]):
    for o,name in enumerate(['subject','template','first']):
     rows.append(dict(template_slot=t,layer=l,origin=name,subjects=n,nonzero_vectors=int(v[t,l,o]),mean_norm=float(np.sqrt(mean_energy[t,l,o])),
      mean_squared_norm=float(total[t,l,o]),trace_variance=float(variation[t,l,o]),
      variation_fraction=float(variation[t,l,o]/total[t,l,o]) if total[t,l,o]>1e-12 else None,
      mean_pairwise_cosine=float(pair[t,l,o]) if v[t,l,o]>1 else None))
  return rows
