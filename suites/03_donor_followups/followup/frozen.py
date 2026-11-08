"""Counterfactual frozen programs. Gates are coefficient fields, not stored facts.
Normalization coefficients and weights are explicitly held fixed unless requested.
"""
import torch
from causal_suite.engine import norm


def copy_program(C):
 return dict(C,layers=[dict(d) for d in C['layers']])


@torch.no_grad()
def realize(engine,C):
 C=copy_program(C);x=C['layers'][0]['x'].clone()
 for l,d in enumerate(C['layers']):
  d['x']=x;d['observed_x']=x+d.get('input_offset',0);x=engine.layer(x,l,C);d['next']=x
 C['z']=engine.read(x,C);return C


def forward_margin(engine,C,g,c):
 x=C['layers'][0]['x']
 for l in range(engine.L):x=engine.layer(x,l,C)
 z=engine.read(x,C,[g,c]);return z[0]-z[1]


def hybrid(engine,recipient,donor,attention=False,gates=False,normalization='recipient'):
 if len(recipient['ids'])!=len(donor['ids']):raise ValueError('Whole-program transplant requires equal token counts')
 C=copy_program(recipient)
 for l,d in enumerate(C['layers']):
  if attention:d['A']=donor['layers'][l]['A'].clone()
  if gates:d['gate']=donor['layers'][l]['gate'].clone()
  if normalization=='donor':
   for key in list(d):
    if key.endswith('layernorm'):d[key]=donor['layers'][l][key].clone()
 if normalization=='donor':C['inv']=donor['inv'].clone()
 elif normalization!='recipient':raise ValueError('Unknown norm policy')
 return realize(engine,C)


def align_positions(recipient,donor):
 """Monotone matching of identical tokens outside the subject, exact subject
 token alignment, and explicit final-position correspondence. Unmatched positions
 retain recipient coefficients. This is partial transport, not a full hybrid.
 """
 from difflib import SequenceMatcher
 R=recipient['input_ids'];D=donor['input_ids'];S=recipient['source'];DS=donor['source']
 if [R[i] for i in S]!=[D[i] for i in DS]:raise ValueError('Paraphrase subject token identities differ')
 mapping=dict(zip(S,DS))
 # Anchors only identical tokens outside subject; final answer site allowed to differ.
 for block in SequenceMatcher(a=R,b=D,autojunk=False).get_matching_blocks():
  for k in range(block.size):
   i,j=block.a+k,block.b+k
   if i not in S and j not in DS:mapping.setdefault(i,j)
 mapping[len(R)-1]=len(D)-1
 # Remove non-monotone or duplicate anchors. Subject anchors are mandatory;
 # conservative single-pass removal of optional anchors preserves causality.
 mandatory=set(S)|{len(R)-1};keep={i:mapping[i] for i in mandatory}
 for i,j in sorted(mapping.items()):
  if i in mandatory:continue
  trial=dict(keep);trial[i]=j;values=[v for _,v in sorted(trial.items())]
  if len(values)==len(set(values)) and all(a<b for a,b in zip(values,values[1:])):keep=trial
 vals=[v for _,v in sorted(keep.items())]
 if len(vals)!=len(set(vals)) or any(a>=b for a,b in zip(vals,vals[1:])):raise ValueError('No monotone subject/final alignment')
 return keep


def aligned_coefficients(C,D,mapping):
 """Return donor desired values + masks in recipient coordinates.
 For attention only causally aligned individual entries are candidates. Rows
 are made valid by the search's mass-preserving edge replacement, not by an
 unacknowledged global renormalization of transplanted entries.
 """
 target=[]
 for r,d in zip(C['layers'],D['layers']):
  gate=r['gate'].clone();gm=torch.zeros_like(gate,dtype=torch.bool)
  A=r['A'].clone();am=torch.zeros_like(A,dtype=torch.bool)
  for i,j in mapping.items():
   gate[i]=d['gate'][j];gm[i]=True
   for k,v in mapping.items():
    if k<=i and v<=j:A[:,i,k]=d['A'][:,j,v];am[:,i,k]=True
  target.append(dict(gate=gate,gate_mask=gm,A=A,A_mask=am))
 return target


def gradients(engine,C,g,c,mode):
 """One reverse sweep computes exact single-coordinate finite-change scores.
 Graph retained within ONE layer at a time; O(L*T*d) cached residuals.
 """
 C=realize(engine,C);x=C['layers'][-1]['next'].detach().requires_grad_(True)
 with torch.enable_grad():r=torch.autograd.grad(engine.read(x,C,[g,c])@x.new_tensor([1.,-1.]),x)[0]
 result=[{} for _ in C['layers']]
 for l in reversed(range(engine.L)):
  temporary=copy_program(C);x=C['layers'][l]['x'].detach().requires_grad_(True);variables=[x];keys=[]
  for key in (['gate'] if mode=='gate' else ['A'] if mode=='edge' else ['gate','A']):
   value=C['layers'][l][key].detach().clone().requires_grad_(True);temporary['layers'][l][key]=value;variables.append(value);keys.append(key)
  with torch.enable_grad():y=engine.layer(x,l,temporary);grads=torch.autograd.grad(y,variables,grad_outputs=r)
  r=grads[0].detach()
  for key,grad in zip(keys,grads[1:]):result[l][key]=grad.detach()
 return C,result


def edge_scores(A,target,gradient):
 # Edit A_j -> desired a, scale other entries by (1-a)/(1-A_j).
 # With other coefficients fixed, the scalar margin is affine in the entire row.
 denom=1-A;weighted=(gradient*A).sum(-1,keepdim=True)
 score=(target-A)*(gradient-(weighted-gradient*A)/denom.clamp_min(1e-8))
 return score.masked_fill(denom<=1e-8,float('-inf'))


def apply_edit(C,edit):
 C=copy_program(C);d=C['layers'][edit['layer']];kind=edit['kind'];index=tuple(edit['index']);value=edit['value']
 if kind=='gate':
  d['gate']=d['gate'].clone();d['gate'][index]=value
 else:
  h,i,j=index;A=d['A'].clone();old=float(A[h,i,j]);den=1-old
  if den<=1e-8:raise ValueError('Cannot redistribute a saturated attention row')
  if not 0<=value<=1:raise ValueError('Invalid donor attention probability')
  A[h,i]*=(1-value)/den;A[h,i,j]=value
  if abs(float(A[h,i].sum())-1)>1e-5:raise AssertionError('Attention simplex violation')
  d['A']=A
 return C


def score_candidates(C,target,grads,used,mode,scope='all'):
 candidates=[]
 for l,(d,t,gr) in enumerate(zip(C['layers'],target,grads)):
  for kind,key,maskkey in [('gate','gate','gate_mask'),('edge','A','A_mask')]:
   if key not in gr:continue
   score=(t[key]-d[key])*gr[key] if kind=='gate' else edge_scores(d[key],t[key],gr[key])
   mask=t[maskkey].clone()
   if scope=='answer':
    restricted=torch.zeros_like(mask)
    if kind=='gate':restricted[-1]=mask[-1]
    else:restricted[:,-1]=mask[:,-1]
    mask=restricted
   for item in used:
    if item[0]==l and item[1]==kind:mask[item[2]]=False
   score=score.masked_fill(~mask,float('-inf'))
   flat=int(score.argmax());best=float(score.flatten()[flat])
   index=[];n=flat
   for size in reversed(score.shape):index.insert(0,n%size);n//=size
   candidates.append(dict(layer=l,kind=kind,index=index,value=float(t[key][tuple(index)]),predicted_gain=best))
 return sorted(candidates,key=lambda e:e['predicted_gain'],reverse=True)


def greedy(engine,C,target,g,c,budget=256,mode='joint',scope='all',prune=True,random_seed=None,tol=1e-4):
 original=C;current=realize(engine,C);initial=float(current['z'][g]-current['z'][c]);trace=[];edits=[];used=set();stop='budget_exhausted'
 import random,math
 rng=random.Random(random_seed)
 def entry(C,n,gain=None,predicted=None,edit=None):
  return dict(edits=n,margin=float(C['z'][g]-C['z'][c]),margin_positive=bool(C['z'][g]>C['z'][c]),gold_top1=int(C['z'].argmax())==g,
    exact_gain=gain,predicted_gain=predicted,edit=edit)
 trace.append(entry(current,0))
 for step in range(budget):
  if trace[-1]['margin_positive']:stop='positive_margin';break
  if random_seed is None:
   current,gr=gradients(engine,current,g,c,mode);choices=score_candidates(current,target,gr,used,mode,scope)
   edit=choices[0] if choices else None
   if edit is None or not math.isfinite(edit['predicted_gain']) or edit['predicted_gain']<=0:stop='no_improving_single_edit';break
  else:
   available=[]
   for l,t in enumerate(target):
    for kind,key in [('gate','gate'),('edge','A')]:
     if mode!='joint' and kind!=mode:continue
     mask=t[key+'_mask'].clone()
     if scope=='answer':
      tmp=torch.zeros_like(mask)
      if kind=='gate':tmp[-1]=mask[-1]
      else:tmp[:,-1]=mask[:,-1]
      mask=tmp
     if kind=='edge':mask &= (1-current['layers'][l]['A'])>1e-8
     for ll,kk,idx in used:
      if ll==l and kk==kind:mask[idx]=False
     nonzero=mask.nonzero()
     if len(nonzero):available.append((l,kind,key,nonzero))
   if not available:stop='no_candidates';break
   pick=rng.randrange(sum(len(x[3]) for x in available))
   for l,kind,key,indices in available:
    if pick<len(indices):
     index=indices[pick].tolist();edit=dict(layer=l,kind=kind,index=index,value=float(target[l][key][tuple(index)]),predicted_gain=None);break
    pick-=len(indices)
  previous=trace[-1]['margin'];current=realize(engine,apply_edit(current,edit));actual=float(current['z'][g]-current['z'][c])-previous
  predicted=edit['predicted_gain']
  if predicted is not None and abs(actual-predicted)>tol*max(1,abs(actual),abs(previous)):
   raise AssertionError(f'Single-edit affine identity failed: predicted {predicted}, actual {actual}')
  edits.append(edit);used.add((edit['layer'],edit['kind'],tuple(edit['index'])));trace.append(entry(current,step+1,actual,predicted,edit))
 if trace[-1]['margin_positive']:stop='positive_margin'
 pruned=list(edits)
 # Deletion-minimal SEQUENCE, not globally minimum cardinality. Edge edits may
 # change previously touched row entries through redistribution: count operations.
 if prune and trace[-1]['margin_positive']:
  changed=True
  while changed:
   changed=False
   for i in range(len(pruned)-1,-1,-1):
    trial=original
    for j,edit in enumerate(pruned):
     if i!=j:trial=apply_edit(trial,edit)
    trial=realize(engine,trial)
    if float(trial['z'][g]-trial['z'][c])>0:pruned.pop(i);changed=True
 return dict(initial_margin=initial,trace=trace,stop=stop,recovered_margin=trace[-1]['margin_positive'],gold_top1=trace[-1]['gold_top1'],
   edit_count=len(edits),pruned_count=len(pruned) if trace[-1]['margin_positive'] else None,pruned_sequence=pruned,
   interpretation='Upper bound under declared donor, alignment and edit units. Not bits, not a global minimum. No-improving-single-edit is not proof that an interacting combination cannot work.')
