import argparse,fcntl,gc,json,signal,time
from pathlib import Path
import numpy as np
import torch
from channel_study.io import load,dump,key,digest,model_load,tokenize,environment
from followup_study.data import selected
from followup_study.features import atomic_npz
from followup_study.mechanisms import capture
from .helpers import reference_rows
from .core import measure,finite_checks,CONDITIONS


def reference(e,tok,rows,tol,out,stop):
 sums=squares=None;n=0;cert=0.
 for r in rows:
  if stop.exists():return None,None
  ids,_,_=tokenize(tok,r,e.device);C=capture(e,ids,tol)
  x=torch.stack([d['observed_x'][-1].detach().cpu().double() for d in C['layers']]);sums=x if sums is None else sums+x;squares=x.square() if squares is None else squares+x.square();n+=1;cert=max(cert,C['certificate']);del C
 mean=sums/n;energy=(squares/n).sum(-1)
 stats=dict(subjects=n,mean_norm=mean.norm(dim=-1).tolist(),mean_squared_norm=energy.tolist(),mean_energy_fraction=(mean.square().sum(-1)/energy.clamp_min(1e-30)).tolist(),native_certificate_max=cert)
 atomic_npz(out/'reference_mean.npz',mean=mean.numpy());dump(out/'reference.json',stats)
 return mean,stats


def main_run(cfg,out,evidence):
 out=Path(out);evidence=Path(evidence);stop=out/'STOP'
 if stop.exists():raise RuntimeError('STOP exists; clear explicitly only after previous worker exits.')
 torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
 if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cohort=load(evidence/'data/cohort.json');refs=reference_rows(cohort,cfg['reference_subjects'],cfg['seed'])
 keys=selected(cohort['evaluation'],cfg['prompts'],cfg['seed']+7);evaluation=[r for r in cohort['evaluation'] if key(r) in keys]
 fdkeys=selected(evaluation,cfg['fd_prompts'],cfg['seed']+19)
 pins={model:load(evidence/model/'model.json') for model in cfg['models']}
 source=Path(__file__).resolve().parents[1];sources={str(p.relative_to(source)):p.read_text() for package in ['projection_study','followup_study','channel_study','causal_suite'] for p in (source/package).glob('*.py')}
 protocol=dict(config=cfg,model_pins=pins,cohort_hash=digest(cohort),source_hash=digest(sources),reference_keys=[key(r) for r in refs],evaluation_keys=[key(r) for r in evaluation],fd_keys=sorted(fdkeys),
  site='block-entry residual, final token',definition='p_S=<d margin/d x_l under freeze S, unit mean residual>. Removed effect=p_live-p_S. Singleton effects overlap.',reference='Same independent calibration subjects as previous anisotropy code at seed0/count100; condition-invariant means within each model.')
 if (out/'protocol.json').exists() and load(out/'protocol.json')!=protocol:raise RuntimeError('Protocol changed; use new output folder.')
 dump(out/'protocol.json',protocol);dump(out/'configuration.json',cfg);dump(out/'environment.json',environment());dump(out/'data/cohort.json',cohort)
 for name,text in sources.items():
  p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 for model in cfg['models']:
  if stop.exists():return
  root=out/model;root.mkdir(parents=True,exist_ok=True);dump(root/'model.json',pins[model])
  # Completed models need no model reload on explicit resume.
  if all((root/'prompts'/key(r)/'measurement.json').exists() for r in evaluation):continue
  print('Loading',model,flush=True);tok,e=model_load(pins[model],cfg['device'])
  if (root/'reference_mean.npz').exists() and (root/'reference.json').exists():
   with np.load(root/'reference_mean.npz',allow_pickle=False) as z:mean=torch.from_numpy(z['mean'].copy())
  else:mean,_=reference(e,tok,refs,cfg['tolerance'],root,stop)
  if mean is None:return
  if tuple(mean.shape)!=(e.L,e.model.config.hidden_size):raise ValueError('Reference shape mismatch')
  for i,r in enumerate(evaluation):
   if stop.exists():return
   folder=root/'prompts'/key(r);path=folder/'measurement.json'
   if path.exists():continue
   ids,_,g=tokenize(tok,r,e.device);C=capture(e,ids,cfg['tolerance']);z=C['z'].clone();z[g]=-torch.inf;c=int(z.argmax())
   result,gradients=measure(e,C,g,c,mean,cfg['baseline_atol'],cfg['baseline_rtol'])
   result.update(model=model,prompt=r,g=g,c=c,finite_checks=finite_checks(e,C,g,c,mean,gradients,cfg) if key(r) in fdkeys else [])
   if cfg['save_gradients']:atomic_npz(folder/'gradients.npz',gradients=torch.stack([gradients[k] for k in CONDITIONS]).numpy(),conditions=np.array(list(CONDITIONS)))
   dump(path,result);del C,gradients;gc.collect();print(f'{model}: {i+1}/{len(evaluation)} projections saved',flush=True)
  del e,tok;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
  from .report import report
  report(out,cfg)
 from .report import report
 report(out,cfg);dump(out/'completion.json',dict(finished=time.time(),models=cfg['models'],prompts_per_model=len(evaluation)))


def run(cfg,out,evidence):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with (out/'run.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  handlers={s:signal.getsignal(s) for s in [signal.SIGINT,signal.SIGTERM]}
  for s in handlers:signal.signal(s,lambda *_:(out/'STOP').touch())
  try:main_run(cfg,out,evidence)
  finally:
   for s,h in handlers.items():signal.signal(s,h)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--evidence',required=True);p.add_argument('--report-only',action='store_true');a=p.parse_args();cfg=load(a.config)
 if a.report_only:
  from .report import report
  report(Path(a.out),cfg)
 else:run(cfg,a.out,a.evidence)
if __name__=='__main__':main()
