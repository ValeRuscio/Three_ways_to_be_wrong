import argparse,json,hashlib,fcntl,gc,platform
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from causal_suite.engine import Engine
from .common import DEFAULT,validate,source,source_hash,dump,evaluation,key
from .cache import capture_case
from .factors import require_swiglu
from .selectors import build,profiles
from . import variance,edits,routing,training,gradients,report,activation


def external_hash(path):
 if not path:return None
 p=Path(path);m=json.loads(p.read_text());files=[p]+[p.parent/x['file'] for x in m.get('bases',[])]+[p.parent/x['file'] for x in m.get('probe_snapshots',[])]
 return {str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def core(src,root,cfg):
 records=src['records'];spec=src['spec'];ev=evaluation(records,cfg);cal=[r for r in records if r.get('eligible') and r['split']=='calibration']
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 dump(root/'cohort.json',dict(evaluation_ids=[str(r['id']) for r in ev],calibration_ids=[str(r['id']) for r in cal],full_evaluation=cfg['max_examples'] is None,
  note='Original eligible fixed cohort. New templates are collected regardless of success; only calibration successes receive detailed factors.'))
 print(f'Loading {spec["repo"]} at {spec["sha"]}',flush=True)
 tok=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['sha']);model=AutoModelForCausalLM.from_pretrained(spec['repo'],revision=spec['sha'],torch_dtype=torch.float32,attn_implementation='eager').to(cfg['device']);engine=Engine(model);require_swiglu(engine)
 total=4*(len(cal)+len(ev));i=0
 for r in cal+ev:
  for t in range(4):
   capture_case(engine,tok,r,t,root,cfg,calibration=r['split']=='calibration');i+=1
   if i%20==0:print(f'{spec["key"]}: {i}/{total} prompt records ready',flush=True)
 index=build(cal,root,cfg)
 if cfg['run_variance']:variance.run(ev,root)
 if cfg['run_profiles']:profiles(ev,root,index,cfg)
 baselines=edits.Baselines(engine,root)
 for i,r in enumerate(ev):
  if cfg['run_activation_patches']:activation.run_case(engine,r,root,index,cfg,baselines)
  if cfg['run_weight_edits']:edits.run_case(engine,tok,r,ev,root,index,cfg,baselines)
  if cfg['run_routing']:routing.run_case(engine,r,cal,root,index,cfg)
  print(f'{spec["key"]}: {i+1}/{len(ev)} intervention cases saved',flush=True)
 report.summarize(root,cfg)
 del engine,model,tok,baselines;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
 dump(root/'CORE_COMPLETE.json',dict(status='complete'))


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--config');p.add_argument('--stage',choices=['all','core','training','report'],default='all');a=p.parse_args()
 cfg=dict(DEFAULT);cfg.update(json.loads(Path(a.config).read_text()) if a.config else {});validate(cfg);out=Path(a.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Another gate-study worker is running in this directory')
 sources=source(a.input,out/'inputs');protocol=dict(config=cfg,input_hashes=[s['hash'] for s in sources],code_hash=source_hash(),gradient_hashes=external_hash(cfg['gradient_manifest']))
 pp=out/'protocol.json'
 if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Code, input or config changed. Use a new output directory.')
 dump(pp,protocol)
 import transformers,numpy
 dump(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,cuda=torch.version.cuda))
 if a.stage!='report':
  if cfg['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in selected existing environment')
  torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for src in sources:
  root=out/src['spec']['key']
  if a.stage in ['all','core']:core(src,root,cfg)
  elif a.stage=='report':report.summarize(root,cfg)
 # The SAME fixed facts are tracked once on OLMo. Do not duplicate trajectories
 # merely because multiple original model result sets were supplied.
 if cfg['run_training'] and a.stage in ['all','training']:
  training.run(sources[0]['records'],out/'olmo_training',cfg);gradients.analyze(cfg['gradient_manifest'],out/'olmo_training',cfg);report.training_plots(out/'olmo_training')
 elif a.stage=='report' and (out/'olmo_training/checkpoints.json').exists():
  training.summarize(out/'olmo_training',cfg);report.training_plots(out/'olmo_training')
 dump(out/f'{a.stage.upper()}_COMPLETE.json',dict(status='complete',stage=a.stage,gradient_alignment='enabled' if cfg['gradient_manifest'] else 'not_run: no actual training gradients supplied'))
 print('Completed:',out,flush=True)

if __name__=='__main__':main()
