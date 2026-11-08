import argparse,gc,json,os,signal,sys,time
from pathlib import Path
import torch
from .core import modules,live
from .io import *
from .experiments import split,fisher,origins,weight_bases,geometry_report,perturb,frontier


def subset(rows,n,seed,one_fact=False):
 selected=sorted(rows,key=lambda r:digest([seed,key(r)]))
 if one_fact:
  unique={}
  for r in selected:unique.setdefault(r['id'],r)
  selected=list(unique.values())
 return selected if n is None else selected[:n]


def _run(cfg,promptfile,out,stages):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);stop=out/'STOP'
 torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
 if torch.cuda.is_available():
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=[json.loads(s) for s in Path(promptfile).read_text().splitlines() if s.strip()]
 rows=[r for r in rows if r['template'] in cfg['templates']]
 rows=subset(rows,cfg['max_prompts'],cfg['seed'])
 if not rows:raise ValueError('No prompts after template filtering')
 source_hash=digest({str(p.relative_to(Path(__file__).parent.parent)):p.read_text() for package in ['channel_study','causal_suite'] for p in (Path(__file__).parent.parent/package).glob('*.py')})
 protocol=dict(config=cfg,prompts_sha256=digest(rows),source_sha256=source_hash,
  regime_sha256=digest(Path(cfg['regimes_csv']).read_text()) if cfg.get('regimes_csv') else None)
 protocolpath=out/'protocol.json'
 if protocolpath.exists() and load(protocolpath)!=protocol:raise ValueError('Protocol changed: use a new output directory')
 dump(protocolpath,protocol);dump(out/'environment.json',environment())
 for package in ['channel_study','causal_suite']:
  for p in (Path(__file__).parent.parent/package).glob('*.py'):
   dest=out/'source'/package/p.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(p.read_text())
 dump(out/'worker.json',dict(pid=os.getpid(),started=time.time(),stages=stages))
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.touch())
 if stop.exists():raise RuntimeError('STOP exists. Remove it explicitly before resuming.')
 regimes=read_regimes(cfg.get('regimes_csv'))
 probe_ids={key(r) for r in subset(rows,cfg['probe_facts'],cfg['seed']+1,True)}
 for spec in cfg['models']:
  if not set(stages)&{'split','fisher','geometry','perturb','frontier'}:break
  root=out/spec['key'];pinned=pin(spec,root/'model.json');tok,e=model_load(pinned,cfg['device'])
  mods=modules(e)
  if len({id(m.weight) for m in mods.values()})!=len(mods):raise ValueError('Shared projection weights are unsupported')
  for i,r in enumerate(rows):
   if stop.exists():print('Stopped at a prompt boundary; saved results retained.',flush=True);return
   started=time.time()
   folder=root/'prompts'/key(r);folder.mkdir(parents=True,exist_ok=True)
   requested=[s for s in stages if s in {'split','fisher','geometry'} or s in {'perturb','frontier'} and key(r) in probe_ids]
   if all((folder/(s+'_DONE.json')).exists() for s in requested) and (folder/'baseline.json').exists():continue
   try:ids,S,g=tokenize(tok,r,e.device)
   except (ValueError,TypeError) as ex:
    dump(folder/'excluded.json',dict(prompt=r,reason=str(ex)));continue
   C=e.capture(ids,cfg['tolerance']);other=C['z'].clone();other[g]=-torch.inf;c=int(other.argmax())
   bp=folder/'baseline.json'
   if bp.exists():base=load(bp)
   else:
    base=dict(prompt=r,g=g,c=c,margin=float(C['z'][g]-C['z'][c]),canonical_top1=bool(C['z'].argmax()==g),
     certificate=C['certificate'],**generate(e,tok,ids,r['aliases'],cfg['max_new_tokens']))
    base['regime']=label(r,base,regimes,spec['key']);dump(bp,base)
   L=F=None
   if any(s in requested and not (folder/(s+'_DONE.json')).exists() for s in ['split','perturb','frontier']):
    summary,L,F=split(e,C,g,c)
    if 'split' in requested:dump(folder/'split.json',summary);dump(folder/'split_DONE.json',dict(status='complete'))
   for stage in requested:
    if (folder/(stage+'_DONE.json')).exists():continue
    if stop.exists():return
    if stage=='fisher':dump(folder/'fisher.json',fisher(e,C,cfg['fisher_draws'],int(digest([cfg['seed'],key(r)])[:8],16)))
    elif stage=='geometry':
     summary,vectors=origins(e,C,S,cfg['tolerance']);dump(folder/'geometry.json',summary)
     temp=folder/'origin_vectors.pt.tmp';torch.save(vectors,temp);temp.replace(folder/'origin_vectors.pt')
    elif stage=='perturb':perturb(e,C,g,c,r,cfg,L,F,folder/'perturb',stop)
    elif stage=='frontier':
     pool=[q for q in rows if q['relation']==r['relation'] and q['group_id']!=r['group_id'] and q['template']==r['template']]
     controls=[];seen=set()
     for q in sorted(pool,key=lambda q:digest([key(r),key(q)])):
      if q['group_id'] in seen:continue
      seen.add(q['group_id']);qi,_,_=tokenize(tok,q,e.device)
      with torch.no_grad():qz=live(e,qi).cpu()
      qb=generate(e,tok,qi,q['aliases'],cfg['max_new_tokens'])
      controls.append(dict(q,ids=qi,z=qz,**qb))
      if len(controls)>=cfg['specificity_n']:break
     if not controls:dump(folder/'frontier_coverage.json',dict(reason='No independent same-relation controls'))
     frontier(e,tok,C,g,c,r,cfg,L,F,controls,folder/'frontier',stop)
     del controls
    if stop.exists():return
    dump(folder/(stage+'_DONE.json'),dict(status='complete'))
   del C,L,F;gc.collect()
   print(f'{spec["key"]}: {i+1}/{len(rows)} prompts saved in {time.time()-started:.1f}s; stages={requested}',flush=True)
  if 'geometry' in stages and not stop.exists():
   paths=sorted((root/'prompts').glob('*/origin_vectors.pt'));bases=weight_bases(e,cfg)
   dump(root/'geometry_subspaces.json',geometry_report(paths,bases,cfg))
  del e,tok;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 if 'training' in stages:
  from .training import run as training_run
  train_rows=subset([r for r in rows if r['template']==cfg['training_template']],cfg['training_facts'],cfg['seed']+2,True)
  if not train_rows:raise ValueError('No training-template prompts')
  for spec in cfg['training_models']:
   if stop.exists():return
   training_run(spec,train_rows,cfg,out/('training_'+spec['key']),stop)
 if stop.exists():return
 from .report import report
 report(out,cfg)
 dump(out/'last_completion.json',dict(stages=stages,time=time.time()))


def run(cfg,promptfile,out,stages):
 import fcntl
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with (out/'run.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Another worker owns this results directory')
  handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
  try:return _run(cfg,promptfile,out,stages)
  finally:
   for sig,h in handlers.items():signal.signal(sig,h)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--prompts',required=True);p.add_argument('--out',required=True)
 p.add_argument('--stages',nargs='+',choices=['split','fisher','geometry','perturb','frontier','training','report'],default=['split','fisher','geometry'])
 a=p.parse_args();cfg=load(a.config)
 if a.stages==['report']:
  from .report import report
  report(Path(a.out),cfg)
 else:run(cfg,a.prompts,a.out,a.stages)
if __name__=='__main__':main()
