import argparse,gc,json,os,signal,time,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
from channel_study.io import load,dump,key,pin,model_load,tokenize,generate,environment,digest
from channel_study.core import modules,stats
from .mechanisms import capture,collect,MECHANISMS,split_rows,three_way,certificates,layer_matrix
from .features import feature_vectors,origin_vectors,Crossed,Moments
from .data import prepare,selected


def base(e,tok,C,r,g,c,cfg):
 return dict(prompt=r,g=g,c=c,margin=float(C['z'][g]-C['z'][c]),canonical_top1=bool(C['z'].argmax()==g),
  native_certificate=C['certificate'],**generate(e,tok,C['ids'],r['aliases'],cfg['max_new_tokens']))


def can_reuse(previous,spec,root):
 if not previous:return None
 p=Path(previous)/spec['key'];mp=p/'model.json'
 if not mp.exists() or not (root/'model.json').exists():return None
 old=load(mp);new=load(root/'model.json')
 return p if old.get('sha')==new.get('sha') and old['repo']==new['repo'] else None


def main_run(cfg,out,stages,previous):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);stop=out/'STOP'
 if stop.exists():raise RuntimeError('STOP exists. Clear it explicitly before resuming.')
 torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
 if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cohort=prepare(cfg,out,previous);ev=cohort['evaluation'];cal=cohort['calibration']
 src=Path(__file__).parents[1]
 sources={str(p.relative_to(src)):p.read_text() for package in ['followup_study','channel_study','causal_suite'] for p in (src/package).glob('*.py')}
 protocol=dict(config=cfg,cohort_hash=digest(cohort),source_hash=digest(sources))
 if (out/'protocol.json').exists() and load(out/'protocol.json')!=protocol:raise ValueError('Protocol changed. Use a new output directory.')
 dump(out/'protocol.json',protocol);dump(out/'configuration.json',cfg);dump(out/'environment.json',environment())
 for name,content in sources.items():
  p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
 sets={stage:selected(ev,cfg[stage+'_prompts'],cfg['seed']+7) for stage in ['null','freeze','matrix','certificate']}
 origin_needed=bool(set(stages)&{'regimes','variance','sink'})
 for spec in cfg['models']:
  active=bool(set(stages)&{'profiles','regimes','variance','sink','certificate'}) or ('nulls' in stages and spec['key'] in cfg['null_models']) or ('freeze' in stages and spec['key'] in cfg['freeze_models']) or ('matrix' in stages and spec['key'] in cfg['matrix_models'])
  if not active:continue
  if stop.exists():return
  root=out/spec['key']
  oldpin=Path(previous)/spec['key']/'model.json' if previous else None
  if oldpin and oldpin.exists() and not (root/'model.json').exists():
   candidate=load(oldpin)
   if candidate['repo']==spec['repo'] and candidate.get('revision','main')==spec.get('revision','main'):dump(root/'model.json',candidate)
  pinned=pin(spec,root/'model.json');tok,e=model_load(pinned,cfg['device']);reuse=can_reuse(previous,spec,root)
  from .mechanisms import norm_modules
  dump(root/'architecture.json',dict(model_type=e.kind,layers=e.L,hidden_size=e.model.config.hidden_size,
   normalization_modules=list(norm_modules(e)),qkv_bias={str(l):{f:getattr(b.self_attn,f+'_proj').bias is not None for f in ['q','k','v']} for l,b in enumerate(e.blocks)}))
  work=cal+ev if 'regimes' in stages else ev
  # Group by fact so three-template ANOVA and sink sufficient statistics can commit atomically per fact.
  grouped={}
  for r in work:grouped.setdefault((r['split'],r['relation'],r['id']),[]).append(r)
  for i,((split,relation,fact),rs) in enumerate(sorted(grouped.items())):
   if stop.exists():return
   started=time.time();relation_key=digest(relation)[:16]
   var=Crossed(root/'variance_state'/f'{relation_key}.npz') if 'variance' in stages and split=='evaluation' else None
   sink=Moments(root/'sink_state'/f'{relation_key}.npz') if 'sink' in stages and split=='evaluation' else None
   subject=rs[0]['group_id']
   # One fact per subject/relation, selected by ID, matching the balanced-design requirement.
   first=min(r['id'] for r in ev if r['group_id']==subject and r['relation']==relation) if split=='evaluation' else fact
   need_var=var is not None and subject not in var.done and fact==first
   need_sink=sink is not None and subject not in sink.done and fact==first
   features={};preacts={}
   for r in sorted(rs,key=lambda r:r['template']):
    if stop.exists():return
    folder=root/'prompts'/key(r);folder.mkdir(parents=True,exist_ok=True)
    is_eval=split=='evaluation';k=key(r)
    todo=[]
    if 'profiles' in stages and is_eval and not (folder/'split.json').exists():todo.append('profiles')
    if 'regimes' in stages and not (folder/'origins.json').exists():todo.append('origins')
    if need_var and r['template'] in cfg['variance_templates']:todo.append('variance')
    if need_sink and r['template'] in cfg['variance_templates']:todo.append('sink')
    for stage,allowed in [('null',cfg['null_models']),('freeze',cfg['freeze_models']),('matrix',cfg['matrix_models']),('certificate',[s['key'] for s in cfg['models']])]:
     if ('nulls' if stage=='null' else stage) in stages:
      if is_eval and spec['key'] in allowed and k in sets[stage] and not (folder/(stage+'.json')).exists():todo.append(stage)
    if 'null' in todo and not (folder/'split.json').exists() and 'profiles' not in todo:todo.insert(0,'profiles')
    if not todo:continue
    try:ids,S,g=tokenize(tok,r,e.device)
    except (ValueError,TypeError) as ex:dump(folder/'excluded.json',dict(prompt=r,reason=str(ex)));continue
    C=capture(e,ids,cfg['tolerance']);other=C['z'].clone();other[g]=-torch.inf;c=int(other.argmax())
    oldfolder=reuse/'prompts'/k if reuse else None
    if (folder/'baseline.json').exists():b=load(folder/'baseline.json')
    elif oldfolder and (oldfolder/'baseline.json').exists():
     b=load(oldfolder/'baseline.json')
     if b['g']!=g or b['c']!=c or abs(b['margin']-float(C['z'][g]-C['z'][c]))>1e-3:raise ValueError('Previous baseline differs')
     dump(folder/'baseline.json',b)
    else:b=base(e,tok,C,r,g,c,cfg);dump(folder/'baseline.json',b)
    b.update(first_token_id=int(ids[0]),first_token_in_subject=0 in S,tokenizer_bos_id=getattr(tok,'bos_token_id',None))
    dump(folder/'baseline.json',b)
    obj=lambda z:z[g]-z[c]
    for stage in todo:
     if stop.exists():return
     if stage in ['freeze','matrix','certificate','null']:print(f'{spec["key"]} prompt {k}: starting {stage}',flush=True)
     if stage=='profiles':
      if oldfolder and (oldfolder/'split.json').exists():
       dump(folder/'split.json',load(oldfolder/'split.json'));dump(folder/'split_provenance.json',dict(reused_from=str(oldfolder),sha=pinned['sha']))
      else:
       L,_=collect(e,C,obj);F,_=collect(e,C,obj,MECHANISMS);dump(folder/'split.json',split_rows(e,L,F));del L,F
     elif stage=='variance':features[r['template']]=feature_vectors(e,C,S)
     elif stage in ['origins','sink']:
      if r['template'] not in preacts:
       origins,vec=origin_vectors(e,C,S,g,c,cfg['tolerance']);dump(folder/'origins.json',origins)
       if r['template'] in cfg['variance_templates']:preacts[r['template']]=vec
     elif stage=='freeze':dump(folder/'freeze.json',three_way(e,C,g,c))
     elif stage=='certificate':
      resolved=sorted({round(v*(e.L-1)) for v in cfg['certificate_layer_fractions']})
      dump(folder/'certificate.json',certificates(e,C,g,c,dict(cfg,certificate_layers_resolved=resolved)))
     elif stage=='matrix':
      result=layer_matrix(e,C,g,c,cfg,stop)
      if result is None:return
      dump(folder/'matrix.json',result)
     elif stage=='null':
      records=[]
      for seed in cfg['readout_seeds']:
       rng=torch.Generator().manual_seed(seed+int(k[:8],16));direction=torch.randn(e.head.weight.shape[1],generator=rng)
       scale=(e.head.weight[g]-e.head.weight[c]).detach().norm().cpu();direction=direction/direction.norm()*scale
       L,_=collect(e,C,obj,readout=direction);F,_=collect(e,C,obj,MECHANISMS,readout=direction)
       records.extend(dict(condition='random_readout',seed=seed,**row) for row in split_rows(e,L,F));del L,F
      dump(folder/'null.json',records)
    del C;gc.collect()
   templates=cfg['variance_templates']
   if need_var and all(t in features for t in templates):
    var.add(subject,{name:[features[t][name] for t in templates] for name in ['gate','attention']})
    dump(root/'variance'/f'{relation_key}.json',dict(relation=relation,rows=var.rows()))
   if need_sink and all(t in preacts for t in templates):
    sink.add(subject,[preacts[t] for t in templates]);dump(root/'sink'/f'{relation_key}.json',dict(relation=relation,rows=sink.rows()))
   print(f'{spec["key"]} {split} fact {i+1}/{len(grouped)} saved ({time.time()-started:.1f}s)',flush=True)
  del e,tok;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
  pending_null=[r for r in ev if key(r) in sets['null'] and (root/'prompts'/key(r)/'baseline.json').exists() and not (root/'prompts'/key(r)/'random_init.json').exists()]
  if 'nulls' in stages and spec['key'] in cfg['null_models'] and pending_null:
   from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
   from causal_suite.engine import Engine
   config=AutoConfig.from_pretrained(pinned['repo'],revision=pinned['sha']);config._attn_implementation='eager'
   torch.manual_seed(cfg['random_initialization_seed']);random_model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32,attn_implementation='eager')
   e=Engine(random_model.to(cfg['device']));tok=AutoTokenizer.from_pretrained(pinned['repo'],revision=pinned['sha'])
   for r in pending_null:
    if stop.exists():return
    k=key(r);folder=root/'prompts'/k
    if k not in sets['null'] or (folder/'random_init.json').exists():continue
    ids,S,g=tokenize(tok,r,e.device);C=capture(e,ids,cfg['tolerance'])
    # Same canonical token AND same competitor as trained counterpart.
    b=load(folder/'baseline.json');c=b['c'];obj=lambda z:z[g]-z[c]
    L,_=collect(e,C,obj);F,_=collect(e,C,obj,MECHANISMS)
    dump(folder/'random_init.json',dict(initialization_seed=cfg['random_initialization_seed'],g=g,c=c,
     certificate=C['certificate'],rows=split_rows(e,L,F)));del C,L,F
    print(f'{spec["key"]} random initialization prompt {k} saved',flush=True)
   del e,tok,random_model;gc.collect()
   if torch.cuda.is_available():torch.cuda.empty_cache()
 if stop.exists():return
 from .report import report
 report(out,cfg);dump(out/'completion.json',dict(stages=stages,time=time.time()))


def run(cfg,out,stages,previous=None):
 import fcntl
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with (out/'run.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Another worker is running')
  handlers={sig:signal.getsignal(sig) for sig in [signal.SIGINT,signal.SIGTERM]}
  for sig in handlers:signal.signal(sig,lambda *_:(out/'STOP').touch())
  dump(out/'worker.json',dict(pid=os.getpid(),stages=stages,started=time.time()))
  try:main_run(cfg,out,stages,previous)
  finally:
   for sig,h in handlers.items():signal.signal(sig,h)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--previous')
 p.add_argument('--stages',nargs='+',choices=['profiles','regimes','variance','sink','nulls','freeze','certificate','matrix','report'],default=['profiles','regimes','variance','sink']);a=p.parse_args();cfg=load(a.config)
 if a.stages==['report']:
  from .report import report
  report(Path(a.out),cfg)
 else:run(cfg,a.out,a.stages,a.previous)
if __name__=='__main__':main()
