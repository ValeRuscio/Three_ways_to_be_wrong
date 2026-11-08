import json,hashlib,gc
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from causal_suite.data import digest
from gate_study.common import dump,npz,load_arrays,table,key,cache_path,meta_path,read_meta


def stamp(*values):return digest(values)[:24]
def load(path):return json.loads(Path(path).read_text())
def layers(e,scope,cfg):
 k=max(1,min(e.L-1,round(e.L*cfg['crossover_fraction'])))
 return list(range(k)) if scope=='below' else list(range(k,e.L)) if scope=='above' else list(range(e.L))

def model_load(spec,cfg):
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from causal_suite.engine import Engine
 tok=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['sha'])
 m=AutoModelForCausalLM.from_pretrained(spec['repo'],revision=spec['sha'],torch_dtype=torch.float32,attn_implementation='eager').to(cfg['device'])
 return tok,Engine(m)

def pin(spec,path):
 from huggingface_hub import HfApi
 if Path(path).exists():return load(path)
 result=dict(spec,sha=HfApi().model_info(spec['repo'],revision=spec.get('revision','main')).sha)
 dump(path,result);return result

def release():
 gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()

def ids(e,m):return torch.tensor(m['input_ids'],device=e.device)

@torch.no_grad()
def logits(e,m,attentions=False):
 x=ids(e,m)
 return e.model(x[None],attention_mask=torch.ones_like(x[None]),use_cache=False,output_attentions=attentions).logits[0,-1]

def endpoint(z,m):
 g=m['g'];c=m['c'];other=z.clone();other[g]=-torch.inf
 return dict(margin=float(z[g]-other.max()),fixed_margin=float(z[g]-z[c]),gold_top1=bool(z.argmax()==g),top_token=int(z.argmax()))

@torch.no_grad()
def generation(e,tok,m,cfg):
 from causal_suite.data import matches
 x=ids(e,m)
 # No KV cache: interventions at fixed prompt sites remain well-defined.
 y=e.model.generate(x[None],attention_mask=torch.ones_like(x[None]),use_cache=False,
  max_new_tokens=cfg['max_new_tokens'],do_sample=False,temperature=1.,top_p=1.,pad_token_id=tok.eos_token_id)
 text=tok.decode(y[0,len(x):],skip_special_tokens=True)
 return dict(generated=text,full_answer_correct=matches(text,m['aliases']))

@lru_cache(maxsize=32)
def measured(root):
 return [load(p) for p in sorted((Path(root)/'features').glob('*/*.json')) if load(p).get('status')=='measured']

def pairs(root,split='evaluation',reverse=False):
 by={}
 for m in measured(root):
  if m['split']==split:by.setdefault(m['id'],[]).append(m)
 out=[]
 for group in by.values():
  # All failing/succeeding template pairs, not a claimed fixed donor count.
  for f in group:
   if f['correct']:continue
   for s in group:
    if not s['correct']:continue
    out.append(dict(recipient=f,donor=s,direction='repair',pair_id=stamp(f['id'],f['template'],s['template'],'repair')))
    if reverse:out.append(dict(recipient=s,donor=f,direction='damage',pair_id=stamp(f['id'],s['template'],f['template'],'damage')))
 return out

def control_metas(root,m,n):
 seen=set();good=[];bad=[]
 for q in sorted(measured(root),key=lambda q:stamp(m['id'],q['id'])):
  if q['split']!='evaluation' or q['template']!=0 or q['relation']!=m['relation'] or q['group_id']==m['group_id'] or q['group_id'] in seen:continue
  seen.add(q['group_id']);(good if q['correct'] else bad).append(q)
 selected=good[:(n+1)//2]+bad[:n//2];taken={q['id'] for q in selected}
 return (selected+[q for q in good+bad if q['id'] not in taken])[:n]

def baseline_controls(e,controls):return [(m,logits(e,m).cpu()) for m in controls]

@torch.no_grad()
def specificity(e,baseline):
 rows=[]
 for m,z0 in baseline:
  z=logits(e,m);z0=z0.to(z.device);b=endpoint(z0,m);a=endpoint(z,m)
  kl=(z0.double().softmax(-1)*(z0.double().log_softmax(-1)-z.double().log_softmax(-1))).sum()
  rows.append(dict(id=m['id'],group_id=m['group_id'],baseline_top1=b['gold_top1'],gold_top1=a['gold_top1'],prediction_changed=a['top_token']!=b['top_token'],margin_gain=a['margin']-b['margin'],kl=float(kl)))
 return rows
