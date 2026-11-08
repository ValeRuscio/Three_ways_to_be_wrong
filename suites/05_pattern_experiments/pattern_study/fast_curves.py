"""Exact causal-prefix reuse for final-prompt-position curve interventions.
No precision change, no different cohort, no dropped controls. Native/frozen
references remain available; every recipient receives equivalence checks.
"""
from contextlib import contextmanager
import time
import torch
from transformers import DynamicCache
from causal_suite.engine import norm
from .common import ids,logits,endpoint,dump
from .interventions import gates,attention,frozen


@contextmanager
def skip_prefix_layers(e,C,start):
 saved=[]
 try:
  for l in range(start):
   b=e.blocks[l];had='forward' in b.__dict__;old=b.__dict__.get('forward');saved.append((b,had,old))
   def forward(hidden_states,*args,l=l,**kwargs):
    if hidden_states.shape[1]!=1:raise ValueError('Cached layers require one final token')
    result=(C['layers'][l]['next'][-1:][None],)
    if kwargs.get('output_attentions',False):result+=(C['layers'][l]['A'][:,-1:][None],)
    return result
   b.forward=forward
  yield
 finally:
  for b,had,old in saved:
   if had:b.forward=old
   else:del b.forward


class FastCurves:
 def __init__(self,e,C,m,tol=1e-4):
  self.e=e;self.C=C;self.m=m;self.tol=tol;self.enabled=False;self.legacy=None;self.values=[]
  if e.kind not in ['llama','qwen2','olmo2']:raise ValueError('Unsupported fast architecture')
  with torch.no_grad():
   # Only earlier-token K/V are cached; intervened final token always recomputes.
   if len(C['ids'])>1:
    x=C['ids'][:-1][None]
    result=e.model(x,attention_mask=torch.ones_like(x),use_cache=True)
    self.legacy=tuple((k.detach(),v.detach()) for k,v in result.past_key_values.to_legacy_cache())
   for l,d in enumerate(C['layers']):
    x=d['x']+d.get('input_offset',0)
    a=x if e.kind=='olmo2' else norm(x,e.blocks[l].input_layernorm,d['input_layernorm'])
    self.values.append(e.values(a,l)[:,:-1].detach())

 def full_live(self,gchanges=None,rows=None):
  with gates(self.e,gchanges or {},len(self.C['ids'])-1),attention(self.e,rows or {}):return logits(self.e,self.m,bool(rows))

 @torch.no_grad()
 def cached_live(self,gchanges=None,rows=None):
  e=self.e;C=self.C;gchanges=gchanges or {};rows=rows or {};start=min(set(gchanges)|set(rows),default=0)
  cache=DynamicCache.from_legacy_cache(self.legacy,num_hidden_layers=e.L) if self.legacy is not None else DynamicCache(num_hidden_layers=e.L)
  hs=[]
  try:
   # Override probabilities, then recompute from LIVE last-token V and immutable
   # prefix V. This differs from the full-sequence intervention only in reuse.
   for l,row in rows.items():
    row=torch.as_tensor(row,device=e.device,dtype=torch.float32);b=e.blocks[l]
    def hook(mod,args,kw,out,l=l,b=b,row=row):
     hidden=args[0] if args else kw['hidden_states'];A=row[:,None,:]
     prefix=self.legacy[l][1][0].repeat_interleave(e.H//e.Hkv,dim=0) if self.legacy else e.values(hidden[0],l)[:,:0]
     V=torch.cat([prefix,e.values(hidden[0],l)],dim=1)
     y=b.self_attn.o_proj((A@V).transpose(0,1).reshape(1,-1))[None]
     parts=list(out);parts[0]=y;parts[1]=A[None];return tuple(parts)
    hs.append(b.self_attn.register_forward_hook(hook,with_kwargs=True))
   with skip_prefix_layers(e,C,start),gates(e,gchanges,0):
    return e.model(C['ids'][-1:][None],attention_mask=torch.ones(1,len(C['ids']),device=e.device,dtype=torch.long),
     past_key_values=cache,use_cache=True,output_attentions=bool(rows),cache_position=torch.tensor([len(C['ids'])-1],device=e.device)).logits[0,-1]
  finally:
   for h in hs:h.remove()

 @torch.no_grad()
 def cached_frozen(self,gchanges=None,rows=None):
  e=self.e;C=self.C;gchanges=gchanges or {};rows=rows or {};start=min(set(gchanges)|set(rows),default=0)
  x=C['layers'][start]['x'][-1:]
  for l in range(start,e.L):
   b=e.blocks[l];d=C['layers'][l];offset=d.get('input_offset');x=x+(offset[-1:] if offset is not None else 0)
   a=x if e.kind=='olmo2' else norm(x,b.input_layernorm,d['input_layernorm'][-1:])
   V=torch.cat([self.values[l],e.values(a,l)],dim=1)
   A=torch.as_tensor(rows[l],device=e.device,dtype=torch.float32)[:,None] if l in rows else d['A'][:,-1:]
   att=b.self_attn.o_proj((A@V).transpose(0,1).reshape(1,-1))
   if e.kind=='olmo2':
    y=x+norm(att,b.post_attention_layernorm,d['post_attention_layernorm'][-1:]);mi=y
   else:
    y=x+att;mi=norm(y,b.post_attention_layernorm,d['post_attention_layernorm'][-1:])
   gate=d['gate'][-1:].clone()
   if l in gchanges:
    ix,v=gchanges[l];gate[0,torch.as_tensor(ix,device=e.device)]=torch.as_tensor(v,device=e.device,dtype=torch.float32)
   mlp=b.mlp.down_proj(gate*b.mlp.up_proj(mi))
   x=y+(norm(mlp,b.post_feedforward_layernorm,d['post_feedforward_layernorm'][-1:]) if e.kind=='olmo2' else mlp)
  return e.read(x,C)

 def check(self,path):
  """Compare native, single-layer and multi-layer edits BEFORE trusting reuse."""
  e=self.e;C=self.C;records=[];nativegate={};att={}
  for l in sorted({0,e.L//2,e.L-1}):
   d=C['layers'][l];width=d['gate'].shape[-1];ix=torch.arange(min(8,width));nativegate[l]=(ix.cpu().numpy(),(-.5*d['gate'][-1,ix.to(e.device)]).cpu().numpy())
   row=d['A'][:,-1].clone();row[0]=row[0].flip(-1);att[l]=row.cpu().numpy()
  probes=[({},{}),( {e.L-1:nativegate[e.L-1]},{}),(nativegate,{}),({},att),(nativegate,att)]
  try:
   for mode in ['live','frozen']:
    for index,(gs,rs) in enumerate(probes):
     full=self.full_live(gs,rs) if mode=='live' else frozen(e,C,gs,rs)
     fast=self.cached_live(gs,rs) if mode=='live' else self.cached_frozen(gs,rs)
     err=float((full-fast).abs().max());relative=float((full-fast).norm()/full.norm().clamp_min(1))
     a=endpoint(full,self.m);b=endpoint(fast,self.m)
     ok=err<=self.tol and relative<=self.tol and a['gold_top1']==b['gold_top1'] and a['top_token']==b['top_token']
     records.append(dict(mode=mode,probe=index,max_logit_error=err,relative_error=relative,passed=ok))
   self.enabled=all(r['passed'] for r in records)
  except (RuntimeError,ValueError,AssertionError) as ex:
   self.enabled=False;records.append(dict(passed=False,error=str(ex)))
  timings={}
  if self.enabled:
   # A timing example, not a promise about the full stage. Warm both paths.
   probe={e.L-1:nativegate[e.L-1]}
   for mode,func in [('full_live',lambda:self.full_live(probe)),('cached_live',lambda:self.cached_live(probe))]:
    func()
    if e.device.type=='cuda':torch.cuda.synchronize(e.device)
    start=time.perf_counter()
    for _ in range(3):func()
    if e.device.type=='cuda':torch.cuda.synchronize(e.device)
    timings[mode+'_seconds']=(time.perf_counter()-start)/3
  dump(path,dict(enabled=self.enabled,checks=records,timing_example=timings,precision='float32; TF32 disabled',fallback='Full original path when validation fails; near-zero gold margins rechecked on full path'))
  return self.enabled

 def evaluate(self,mode,gchanges=None,rows=None):
  if self.enabled:
   z=self.cached_live(gchanges,rows) if mode=='live' else self.cached_frozen(gchanges,rows)
   # Protect discrete gold-top1 endpoint at numerical decision boundaries.
   if abs(endpoint(z,self.m)['margin'])>10*self.tol:return z
  return self.full_live(gchanges,rows) if mode=='live' else frozen(self.e,self.C,gchanges,rows)
