import argparse,fcntl,hashlib,json,platform,shutil
from pathlib import Path
import torch
from .config import DEFAULT,validate
from .common import *
from . import capture,analysis,weights,curves,sink,training,report


def source_records(path):
 """Read ONLY JSON data from original archive, never extract/execute archive code."""
 import zipfile
 path=Path(path).expanduser()
 if path.is_file() and path.suffix=='.zip':
  with zipfile.ZipFile(path) as z:
   names=sorted(n for n in z.namelist() if n.endswith('/raw/measurements.json'))
   if not names:raise ValueError('Use original causal_v1_results.zip, which contains facts/aliases/splits. Gate/followup compact ZIPs are not the cohort input.')
   raw=z.read(names[0]);records=json.loads(raw)
 else:
  if path.is_dir():
   names=sorted(path.glob('**/raw/measurements.json'))
   if not names:raise ValueError('No original raw/measurements.json found')
   path=names[0]
  raw=path.read_bytes();records=[json.loads(s) for s in raw.decode().splitlines() if s.strip()] if path.suffix=='.jsonl' else json.loads(raw)
 rows=[];excluded=[];seen={};subjects={}
 for r in records:
  try:
   if r['split'] not in ['calibration','evaluation']:continue
   if not all(isinstance(r.get(k),str) and r[k].strip() for k in ['subject','answer','relation']):raise ValueError('missing text')
   if not isinstance(r.get('aliases'),list) or not r['aliases']:raise ValueError('missing aliases')
   q={k:r[k] for k in ['id','group_id','subject','answer','relation','aliases','split']};q['id']=str(q['id']);q['group_id']=str(q['group_id'])
   if q['id'] in seen:raise ValueError('Duplicate fact id')
   if q['group_id'] in subjects and subjects[q['group_id']]!=q['split']:raise ValueError('Subject overlaps calibration and evaluation')
   seen[q['id']]=True;subjects[q['group_id']]=q['split'];rows.append(q)
  except (KeyError,ValueError) as ex:
   if 'overlaps' in str(ex) or 'Duplicate' in str(ex):raise
   excluded.append(dict(id=r.get('id'),reason=str(ex)))
 if not rows:raise ValueError('No valid input facts')
 return rows,excluded,hashlib.sha256(raw).hexdigest()


def code_hash():
 root=Path(__file__).parent.parent
 return stamp({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['pattern_study','gate_study','causal_suite'] for p in sorted((root/d).glob('*.py'))})


def core(spec,records,root,cfg):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 done=root/'CORE_COMPLETE.json'
 if done.exists() and set(cfg['stages'])-{'training'} <= set(load(done).get('stages',[])):
  print('Reusing completed model stages:',spec['key'],flush=True);return
 spec=pin(spec,root/'model.json')
 active=set(cfg['stages']);needs_gpu=active&{'capture','weights','curves','sink'}
 if 'capture' not in active and not list((root/'features').glob('*/*.json')):raise ValueError('Run capture first for '+spec['key'])
 if needs_gpu:
  tok,e=model_load(spec,cfg)
  dump(root/'architecture.json',dict(layers=e.L,heads=e.H,hidden=e.model.config.hidden_size,intermediate=e.model.config.intermediate_size,scope_boundary=round(e.L*cfg['crossover_fraction']),boundary_note='Fixed normalized boundary, not fitted to this model outcomes'))
  if 'capture' in active:capture.run(e,tok,records,root,cfg)
  measured.cache_clear()
  if 'analysis' in active:
   if len(active)>1 and (root/'reports/interpretation.json').exists():print('Reusing completed cached analyses:',spec['key'],flush=True)
   else:analysis.run(root,cfg)
  if 'weights' in active:weights.run(e,tok,root,cfg)
  if 'curves' in active:curves.run(e,root,cfg)
  if 'sink' in active:sink.run(e,root,cfg)
  del tok,e;release()
 elif 'analysis' in active:
   if len(active)>1 and (root/'reports/interpretation.json').exists():print('Reusing completed cached analyses:',spec['key'],flush=True)
   else:analysis.run(root,cfg)
 report.run(root,cfg);dump(root/'CORE_COMPLETE.json',dict(status='complete',stages=cfg['stages']))


def cross_model(out,cfg):
 import pandas as pd
 rows=[];tokens=[];facts={}
 for spec in cfg['models']:
  root=out/spec['key'];p=root/'reports/variance_components.csv'
  if p.exists():
   v=pd.read_csv(p);L=int(v.layer.max())+1
   for measurement,s in v.groupby('measurement'):
    a=s.groupby('layer')[['subject_ss','template_ss','total_ss']].sum();dif=a.subject_ss-a.template_ss
    crossing=[int(l) for l in a.index[1:] if dif.loc[l-1]<=0<dif.loc[l]]
    rows.append(dict(model=spec['key'],measurement=measurement,layers=L,first_subject_dominant_layer=crossing[0] if crossing else None,depth_fraction=(crossing[0]+1)/L if crossing else None))
  p=root/'tokenizer.json'
  if p.exists():tokens.append(dict(model=spec['key'],**load(p)))
  facts[spec['key']]={m['id'] for m in measured(root) if m['split']=='evaluation' and m['template']==0}
 table(rows,out/'cross_model_crossovers.csv');table(tokens,out/'tokenizer_comparison.csv')
 common=sorted(set.intersection(*facts.values())) if facts else []
 dump(out/'common_fact_coverage.json',dict(common_evaluation_ids=common,per_model={k:len(v) for k,v in facts.items()},note='Common facts, native tokenizers. Token IDs and gate coordinates are not pooled across architectures.'))


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--config');p.add_argument('--report-only',action='store_true');p.add_argument('--upgrade-fast',action='store_true');p.add_argument('--only',choices=['capture','analysis','weights','curves','sink','training']);a=p.parse_args()
 cfg=dict(DEFAULT);cfg.update(load(a.config) if a.config else {});validate(cfg);out=Path(a.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('A worker already holds this output directory')
 rows,excluded,input_hash=source_records(a.input)
 if cfg['max_eval'] is not None:
  ev=sorted([r for r in rows if r['split']=='evaluation'],key=lambda r:stamp(r['id']))[:cfg['max_eval']];rows=[r for r in rows if r['split']=='calibration']+ev
 protocol=dict(config=cfg,input_hash=input_hash,code_hash=code_hash())
 if (out/'protocol.json').exists() and load(out/'protocol.json')!=protocol:
  if a.upgrade_fast:
   from .upgrade import migrate
   migrate(out,protocol,Path(__file__).parent.parent)
  else:raise ValueError('Input/code/config changed. Use --upgrade-fast only to migrate the exact original run with unchanged settings.')
 dump(out/'protocol.json',protocol);dump(out/'cohort.json',dict(records=rows,excluded=excluded))
 source=out/'source'
 if not source.exists():
  for name in ['pattern_study','gate_study','causal_suite']:shutil.copytree(Path(__file__).parent.parent/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
 import transformers
 dump(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda))
 if not a.report_only:
  if transformers.__version__!='4.48.3':raise RuntimeError('Use existing transformers 4.48.3 environment; this engine has been verified against that interface')
  if cfg['device'].startswith('cuda') and a.only!='analysis' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in the selected environment')
  torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 execution=dict(cfg,stages=[a.only]) if a.only else cfg
 for spec in cfg['models']:
  if a.only=='training':continue
  if Path(spec['key']).name!=spec['key']:raise ValueError('Invalid model key')
  if a.report_only:report.run(out/spec['key'],cfg)
  else:core(spec,rows,out/spec['key'],execution)
 cross_model(out,cfg)
 if not a.report_only and 'training' in execution['stages']:
  for spec in cfg['training_models']:training.run(spec,out/('training_'+spec['key']),cfg)
 dump(out/'REPORT_COMPLETE.json' if a.report_only else out/((a.only.upper() if a.only else 'ALL')+'_COMPLETE.json'),dict(status='complete',models=cfg['models'],stages=cfg['stages']))
 print('Completed:',out,flush=True)

if __name__=='__main__':main()
