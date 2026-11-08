import torch
from channel_study.io import digest,key
from channel_study.core import live
from followup_study.mechanisms import freezes

def cosine(a,b):
 a=a.double();b=b.double();den=float(a.norm()*b.norm())
 return float(torch.dot(a,b)/den) if den>1e-15 else None

def residual_gradients(e,C,g,c,which=()):
 """d margin / d x_l at final token of block-entry residual stream."""
 grads={};handles=[]
 with freezes(e,C,which):
  try:
   handles.append(e.model.get_input_embeddings().register_forward_hook(lambda m,a,out:out.detach().requires_grad_(True)))
   for l,b in enumerate(e.blocks):
    def hook(mod,args,kw,l=l):
     x=args[0] if args else kw['hidden_states']
     def save(d):grads[l]=d[0,-1].detach().cpu().clone()
     if x.requires_grad:x.register_hook(save)
    handles.append(b.register_forward_pre_hook(hook,with_kwargs=True))
   z=live(e,C['ids']);loss=z[g]-z[c];loss.backward();value=float(loss.detach())
  finally:
   for h in handles:h.remove()
 if set(grads)!=set(range(e.L)):raise AssertionError('Missing residual gradients')
 return torch.stack([grads[l] for l in range(e.L)]),value

def reference_rows(cohort,count,seed):
 evaluation={str(r['group_id']) for r in cohort['evaluation']}
 eligible=[r for r in cohort['calibration'] if str(r['group_id']) not in evaluation]
 # One prompt per independent subject, no correctness selection.
 by={}
 for r in sorted(eligible,key=lambda r:digest([seed,key(r)])):by.setdefault(str(r['group_id']),r)
 rows=list(by.values())[:count]
 if len(rows)<2:raise ValueError('Need independent reference subjects')
 return rows
