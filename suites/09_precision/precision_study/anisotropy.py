"""Fresh residual-space gradient captures; no invalid weight-matrix/vector cosine."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from channel_study.io import dump,load,key,model_load,tokenize,digest,environment
from channel_study.core import live,modules,dot
from channel_study.report import summarize
from followup_study.mechanisms import capture,freezes,MECHANISMS,collect
from followup_study.data import selected


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


def output_projection(factors,mean):
 """Frobenius projection onto matrices with output direction parallel to mean.
 This is a nonnegative subspace cosine, not a signed vector cosine.
 """
 if factors is None or float(mean.norm())<=1e-15:return None
 den=dot(factors,factors)
 if den<=1e-30:return None
 d,x=factors;u=mean.double()/mean.double().norm()
 row=(d.double()@u)@x.double()
 return float((row.square().sum()/den).clamp(0,1).sqrt())


def weight_projections(e,C,g,c,mean):
 names=[n for n in modules(e) if n.split(':')[1] in ['down','o']]
 obj=lambda z:z[g]-z[c];L,_=collect(e,C,obj,selected=names);F,_=collect(e,C,obj,MECHANISMS,selected=names)
 rows=[]
 for name in names:
  l,f=name.split(':');l=int(l)
  rows.append(dict(layer=l,family=f,content_output_mean_projection=output_projection(F.get(name),mean[l]),live_output_mean_projection=output_projection(L.get(name),mean[l])))
 return rows


def reference_rows(cohort,count,seed):
 evaluation={str(r['group_id']) for r in cohort['evaluation']}
 eligible=[r for r in cohort['calibration'] if str(r['group_id']) not in evaluation]
 # One prompt per independent subject, no correctness selection.
 by={}
 for r in sorted(eligible,key=lambda r:digest([seed,key(r)])):by.setdefault(str(r['group_id']),r)
 rows=list(by.values())[:count]
 if len(rows)<2:raise ValueError('Need independent reference subjects')
 return rows


def references(e,tok,rows,tolerance):
 sums=None;squares=None;n=0;cert=0.
 for r in rows:
  ids,_,_=tokenize(tok,r,e.device);C=capture(e,ids,tolerance)
  X=torch.stack([d['observed_x'][-1].detach().cpu().double() for d in C['layers']])
  sums=X if sums is None else sums+X;squares=X.square() if squares is None else squares+X.square();n+=1
  cert=max(cert,C['certificate']);del C
 mean=sums/n;mean_norm=mean.norm(dim=-1);energy=(squares/n).sum(-1)
 return mean,dict(subjects=n,mean_norm=mean_norm.tolist(),mean_squared_norm=energy.tolist(),mean_energy_fraction=(mean_norm.square()/energy.clamp_min(1e-30)).tolist(),native_certificate_max=cert)


def measure(e,C,g,c,mean):
 L,lv=residual_gradients(e,C,g,c);F,fv=residual_gradients(e,C,g,c,MECHANISMS)
 if abs(lv-fv)>1e-3*max(1,abs(lv)):raise AssertionError('Frozen baseline mismatch')
 S=L-F;rows=[]
 for l in range(e.L):
  x=C['layers'][l]['observed_x'][-1].detach().cpu();a=L[l].double();b=F[l].double();s=a-b
  aa=float(a@a);bb=float(b@b);ss=float(s@s);cross=float(b@s)
  rows.append(dict(layer=l,content_mean_cosine=cosine(b,mean[l]),live_mean_cosine=cosine(a,mean[l]),selection_mean_cosine=cosine(s,mean[l]),
   content_local_residual_cosine=cosine(b,x),live_local_residual_cosine=cosine(a,x),
   content_selection_cosine=cosine(b,s),content_norm=float(b.norm()),selection_norm=float(s.norm()),
   closure_error=abs(aa-bb-ss-2*cross)/max(1,aa),baseline_disagreement=abs(lv-fv)))
 return rows


def run(cfg,out,evidence,stop_root=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);evidence=Path(evidence)
 cohort=load(evidence/'data/cohort.json');refs=reference_rows(cohort,cfg['anisotropy_reference_subjects'],cfg['seed'])
 ids=selected(cohort['evaluation'],cfg['anisotropy_prompts'],cfg['seed']+7)
 evaluation=[r for r in cohort['evaluation'] if key(r) in ids]
 stop=lambda:(out/'STOP').exists() or (stop_root is not None and (Path(stop_root)/'STOP').exists())
 torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
 if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False
 dump(out/'environment.json',environment());dump(out/'configuration.json',cfg)
 dump(out/'protocol.json',dict(site='block-entry residual, final prompt token',objective='canonical gold versus trained top non-gold token, fixed in initialized counterpart',
  reference_subjects=[r['group_id'] for r in refs],reference_prompt_keys=[key(r) for r in refs],evaluation_prompt_keys=[key(r) for r in evaluation],
  limitation='Residual-space derivative, not a weight-matrix gradient direction. A mean-direction cosine is descriptive; it alone cannot prove normalization explains opposition. Reference means are condition-specific, independent of evaluation subjects.',source_sha=digest(Path(__file__).read_text())))
 allrows=[];trained_ids={}
 spec=next(m for m in cfg['models'] if m['key']=='llama1b');pinned=load(evidence/'llama1b/model.json');dump(out/'llama1b/model.json',pinned)
 from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
 from causal_suite.engine import Engine
 for condition,seed in [('trained',cfg['seed'])]+[('initialized',s) for s in cfg['anisotropy_init_seeds']]:
  if stop():return
  if condition=='trained':tok,e=model_load(pinned,cfg['device'])
  else:
   config=AutoConfig.from_pretrained(pinned['repo'],revision=pinned['sha']);config._attn_implementation='eager'
   torch.manual_seed(seed);model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32,attn_implementation='eager');e=Engine(model.to(cfg['device']));tok=AutoTokenizer.from_pretrained(pinned['repo'],revision=pinned['sha'])
  mean,refstats=references(e,tok,refs,cfg['tolerance']);root=out/f'{condition}_{seed}';root.mkdir(exist_ok=True)
  dump(root/'reference.json',refstats)
  np.savez_compressed(root/'reference_mean.npz',mean=mean.numpy())
  for i,r in enumerate(evaluation):
   if stop():return
   ids,_,g=tokenize(tok,r,e.device);C=capture(e,ids,cfg['tolerance']);k=key(r)
   if condition=='trained':
    z=C['z'].clone();z[g]=-torch.inf;c=int(z.argmax());trained_ids[k]=(g,c)
   else:
    oldg,c=trained_ids[k]
    if oldg!=g:raise AssertionError('Token target changed')
   rows=measure(e,C,g,c,mean)
   records=[dict(model='llama1b',condition=condition,seed=seed,id=r['id'],group_id=r['group_id'],template=r['template'],relation=r['relation'],g=g,c=c,**row) for row in rows]
   weights=[dict(model='llama1b',condition=condition,seed=seed,id=r['id'],group_id=r['group_id'],template=r['template'],relation=r['relation'],**row) for row in weight_projections(e,C,g,c,mean)]
   dump(root/(k+'.json'),dict(prompt=r,rows=records,weight_rows=weights,native_certificate=C['certificate']))
   allrows.extend(records);del C;gc.collect()
   print(f'Anisotropy {condition}/{seed}: {i+1}/{len(evaluation)} saved',flush=True)
  del e,tok
  if condition=='initialized':del model
  gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 report(out,cfg,allrows);dump(out/'completion.json',dict(time=time.time(),prompts=len(evaluation),conditions=1+len(cfg['anisotropy_init_seeds'])))


def report(out,cfg,rows=None):
 out=Path(out);dest=out/'reports';dest.mkdir(exist_ok=True)
 if rows is None:rows=[r for p in out.glob('*/*.json') if p.name!='reference.json' for r in load(p).get('rows',[])]
 if not rows:return
 df=pd.DataFrame(rows);df.to_csv(dest/'residual_anisotropy_per_prompt.csv',index=False)
 metrics=['content_mean_cosine','live_mean_cosine','selection_mean_cosine','content_local_residual_cosine','live_local_residual_cosine','content_selection_cosine','content_norm','selection_norm']
 # Keep seeds explicit; do not present many prompts as many random initializations.
 summary=summarize(df,['model','condition','seed','layer'],metrics,cfg);summary.to_csv(dest/'residual_anisotropy_summary.csv',index=False)
 weights=[r for p in out.glob('*/*.json') if p.name!='reference.json' for r in load(p).get('weight_rows',[])]
 if weights:
  w=pd.DataFrame(weights);w.to_csv(dest/'weight_output_projection_per_prompt.csv',index=False)
  summarize(w,['model','condition','seed','layer','family'],['content_output_mean_projection','live_output_mean_projection'],cfg).to_csv(dest/'weight_output_projection_summary.csv',index=False)
 joins=['model','id','group_id','template','relation','layer'];trained=df[df.condition=='trained'];paired=[]
 for seed,random in df[df.condition=='initialized'].groupby('seed'):
  d=trained.merge(random,on=joins,suffixes=('_trained','_initialized'),validate='one_to_one')
  for col in metrics:d[col+'_difference']=d[col+'_trained']-d[col+'_initialized']
  d['initialization_seed']=seed;paired.append(d)
 if paired:
  p=pd.concat(paired);summarize(p,['model','initialization_seed','layer'],[m+'_difference' for m in metrics],cfg).to_csv(dest/'paired_anisotropy_summary.csv',index=False)
 dump(dest/'checks.json',dict(max_closure=float(df.closure_error.max()),max_baseline_disagreement=float(df.baseline_disagreement.max()),interpretation='Reference means estimated separately within trained and initialized conditions; CIs condition on these reference sets and initialization seeds. Directional, not causal, evidence.'))


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--evidence',required=True);p.add_argument('--stop-root');a=p.parse_args();run(load(a.config),a.out,a.evidence,a.stop_root)
if __name__=='__main__':main()
