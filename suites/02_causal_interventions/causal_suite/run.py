"""python -m causal_suite.run --config config.json --out results/causal_v1"""
import argparse,json,gc,re,os,time
from pathlib import Path
import numpy as np
import torch
from .data import save,digest,build_fixed,inventory,render
from .engine import Engine
from .measure import measure,assign_labels
from .repairs import donors,execute_grid,planted


def resolve_models(cfg,out):
 from huggingface_hub import HfApi
 api=HfApi();path=out/'resolved_models.json'
 if path.exists():return json.loads(path.read_text())
 specs=list(cfg['models'])
 if cfg.get('discover_olmo_training'):
  for family,repo in [('olmo1_training','allenai/OLMo-2-0425-1B'),('olmo7_training','allenai/OLMo-2-1124-7B')]:
   refs=api.list_repo_refs(repo)
   # Stage 1 only: mixing stage-local step counters would create a false timeline.
   branches=[]
   for b in refs.branches:
    match=re.search(r'^stage1-step(\d+)',b.name)
    if match and int(match.group(1))>0:branches.append((int(match.group(1)),b.name))
   branches.sort()
   if not branches:raise RuntimeError(f'No stage1 HF branches found for {repo}. Specify verified revisions explicitly; refusing fabricated checkpoints.')
   ix=sorted(set(np.linspace(0,len(branches)-1,min(7,len(branches))).round().astype(int)))
   from .config import spec
   specs += [spec(family+'_'+str(branches[i][0]),repo,family,'checkpoint',branches[i][1],branches[i][0],False) for i in ix]
 for index,s in enumerate(specs):
  s=dict(s);s['sha']=api.model_info(s['repo'],revision=s['revision']).sha
  # Mutate exact list entry for durable metadata.
  specs[index]=s
 if len({s['key'] for s in specs})!=len(specs):raise ValueError('Model keys must be unique')
 save(path,specs);return specs


def run(cfg,out):
 from transformers import AutoModelForCausalLM,AutoTokenizer
 import transformers
 if transformers.__version__!='4.48.3':raise RuntimeError('Use transformers==4.48.3 and restart kernel')
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 import fcntl
 run_lock=(out/'.run.lock').open('a')
 try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Another worker holds this output directory')
 code={p.name:p.read_text() for p in Path(__file__).parent.glob('*.py')};fingerprint=digest(code)
 inputs={name:digest(Path(cfg[name]).read_text()) for name in ['facts_jsonl','exposure_csv'] if cfg.get(name)}
 state={'config':cfg,'code_hash':fingerprint,'input_hashes':inputs}
 if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=state:raise RuntimeError('Configuration or code changed: choose a NEW output directory. Existing results are preserved.')
 save(out/'protocol.json',state)
 torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4)
 if cfg['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required by this configuration')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 fixed=build_fixed(cfg,out);facts=fixed['rows'];pools=inventory(facts);specs=resolve_models(cfg,out)
 modes=['raw','chat'] if cfg['chat_check'] else ['raw']
 for s in specs:
  tok=AutoTokenizer.from_pretrained(s['repo'],revision=s['sha'],use_fast=True)
  if not tok.is_fast:raise ValueError('Fast tokenizer needed for exact subject spans')
  vocab_hash=digest(tok.get_vocab())
  family_file=out/'shared_decoys'/s['family']/'tokenizer.json'
  if family_file.exists() and json.loads(family_file.read_text())['vocab_hash']!=vocab_hash:raise RuntimeError('Paired models have different vocabularies. Use a new family and compare answer strings; token-ID sharing would be invalid.')
  save(family_file,{'vocab_hash':vocab_hash})
  chat_tok=None
  if cfg['chat_check']:
   reference=next((a for a in specs if a['family']==s['family'] and a['stage']=='instruct'),s)
   chat_tok=AutoTokenizer.from_pretrained(reference['repo'],revision=reference['sha'])
   if not chat_tok.chat_template:raise RuntimeError('No chat template for secondary check; set chat_check=False or choose a paired instruct model')
  print(f'Loading {s["key"]}: {s["repo"]}@{s["sha"]}',flush=True)
  model=AutoModelForCausalLM.from_pretrained(s['repo'],revision=s['sha'],torch_dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).to(cfg['device'])
  engine=Engine(model)
  for mode in modes:
   directory=out/s['key']/mode;directory.mkdir(parents=True,exist_ok=True);save(directory/'model.json',s)
   renderer=chat_tok if mode=='chat' else None
   measured=[]
   invalid_subjects=[dict(id=r['id'],group_id=r['group_id'],split=r['split'],relation=r['relation'],reason='missing_or_empty_subject') for r in facts if r['split']!='inventory' and (not isinstance(r.get('subject'),str) or not r['subject'].strip())]
   save(directory/'invalid_data_rows.json',dict(excluded=invalid_subjects,count=len(invalid_subjects),policy='Excluded before generation and from accuracy denominators; no replacement sampling. Inventory and decoy pools unchanged.'))
   for i,row in enumerate(facts):
    if row['split']=='inventory':continue
    if not isinstance(row.get('subject'),str) or not row['subject'].strip():continue
    path=directory/'records'/f'{digest(row["id"])}.json';hard_path=out/'shared_decoys'/s['family']/mode/f'{digest(row["id"])}.json'
    if path.exists():r=json.loads(path.read_text())
    else:
     prompt=render(row,chat_tokenizer=renderer);hard=json.loads(hard_path.read_text()) if hard_path.exists() else None
     started=time.time();r,C=measure(engine,tok,row,prompt,pools[row['relation']],hard,cfg,mode=='chat');r['seconds']=time.time()-started
     if 'hard_tokens' in r and not hard_path.exists():save(hard_path,r['hard_tokens'])
     if cfg['save_frozen_cache'] and C is not None:
      cache_dir=directory/'frozen_cache';cache_dir.mkdir(exist_ok=True)
      def cpu(x):return x.cpu() if isinstance(x,torch.Tensor) else {k:cpu(v) for k,v in x.items()} if isinstance(x,dict) else [cpu(v) for v in x] if isinstance(x,list) else x
      torch.save(cpu(C),cache_dir/f'{digest(row["id"])}.pt')
     save(path,r);del C
    measured.append(r)
    if i%10==0:print(f'{s["key"]}/{mode}: {len(measured)} fixed prompts measured',flush=True)
   measured=assign_labels(measured,cfg,s['step'] is not None);save(directory/'measurements.json',measured)
   if cfg['run_repairs'] and s['repairs']:
    evaluation=[r for r in measured if r['split']=='evaluation' and r.get('eligible')]
    evaluation.sort(key=lambda r:digest([cfg['seed'],r['id']]))
    if cfg['repair_limit'] is not None:evaluation=evaluation[:cfg['repair_limit']]
    for i,r in enumerate(evaluation):
     path=directory/'repair_records'/f'{digest(r["id"])}.json'
     if path.exists():continue
     C=engine.capture(torch.tensor(r['input_ids'],device=engine.device));D,W,audit=donors(engine,tok,r,measured,cfg,renderer)
     rows=execute_grid(engine,r,C,cfg,D,W);save(path,dict(rows=rows,donor_available=D is not None,wrong_donor_available=W is not None,id=r['id'],attempts=audit))
     print(f'{s["key"]}/{mode}: repairs {i+1}/{len(evaluation)}, donor={D is not None}',flush=True)
     del C,D,W
    if cfg['run_planted']:
     clean=[r for r in evaluation if r['correct']][:cfg['plant_limit']]
     for i,r in enumerate(clean):
      path=directory/'plant_records'/f'{digest(r["id"])}.json'
      if path.exists():continue
      C=engine.capture(torch.tensor(r['input_ids'],device=engine.device));rows,audit=planted(engine,r,C,cfg);save(path,dict(rows=rows,audit=audit));del C
      print(f'{s["key"]}/{mode}: planted controls {i+1}/{len(clean)}',flush=True)
   save(directory/'status.json',dict(status='complete',measured=len(measured),evaluation=sum(r['split']=='evaluation' for r in measured)))
  del engine,model;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 from .report import report
 report(out,cfg)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--report-only',action='store_true');a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
 if a.report_only:
  from .report import report
  report(Path(a.out),cfg)
 else:run(cfg,a.out)
if __name__=='__main__':main()
