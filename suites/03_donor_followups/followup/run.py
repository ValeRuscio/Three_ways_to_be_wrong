"""Resumable follow-up runner; never executes code from an input results ZIP."""
import argparse,json,hashlib,fcntl,os,platform
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from causal_suite.engine import Engine
from causal_suite.data import digest
from .common import DEFAULT,sources,dump,group_label,regime
from .cpu import posthoc,matched_repairs
from .frozen import hybrid,aligned_coefficients,greedy
from .donors import layout_donors,paraphrase_donor
from .sink import vector_snapshot,frequency_vector,aggregate_snapshots
from .geometry import load_manifest,analyze
from .frequency import tokenizer_hash
from .report import report


def code_hash():
 root=Path(__file__).parent.parent
 return digest({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for name in ['followup','causal_suite'] for p in sorted((root/name).glob('*.py'))})


def external_hashes(cfg):
 result={}
 values=[(k,cfg.get(k)) for k in ['frequency_file','subspace_manifest']]+[('frequency_file',v) for v in cfg.get('frequency_files',{}).values()]
 for key,value in values:
  if not value:continue
  p=Path(value).expanduser().resolve();result[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  if key=='frequency_file':files=[p.with_suffix('.json')]
  else:files=[p.parent/r['file'] for r in json.loads(p.read_text())['heads']]
  for f in files:result[str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest()
 return result


def validate(cfg):
 unknown=set(cfg)-set(DEFAULT)
 if unknown:raise ValueError(f'Unknown configuration keys: {unknown}')
 if cfg['edit_scope'] not in ['all','answer'] or any(m not in ['gate','edge','joint'] for m in cfg['edit_modes']):raise ValueError('Invalid edit mode/scope')
 if cfg['edit_budget']<1 or not 0<cfg['threshold']<1:raise ValueError('Invalid budget/threshold')
 if cfg['max_examples'] is not None and cfg['max_examples']<1:raise ValueError('max_examples must be positive or null')


def metadata(r,mode):
 return dict(id=r['id'],group_id=r['group_id'],relation=r['relation'],label=group_label(r),regime=regime(r),mode=mode,
  template_key=digest(r['prompt'].replace(r['subject'],'{SUBJECT}')))


def hybrid_rows(engine,C,r,records,cfg,mode):
 donors=layout_donors(r,records,cfg);rows=[]
 for donor_type,selected in donors.items():
  for donor in selected:
   D=engine.capture(torch.tensor(donor['input_ids'],device=engine.device))
   for policy in cfg['hybrid_norms']:
    for A in [False,True]:
     for G in [False,True]:
      H=hybrid(engine,C,D,A,G,policy);ledger=engine.ledger(H,r['source'],[r['g'],r['c']],[],cfg['certificate_tolerance'])
      origins={key:ledger[str(r['g'])]['origins'][key]-ledger[str(r['c'])]['origins'][key] for key in ledger[str(r['g'])]['origins']}
      margin=float(H['z'][r['g']]-H['z'][r['c']]);error=abs(sum(origins.values())-margin)/max(1,abs(margin))
      if error>cfg['certificate_tolerance']:raise AssertionError('Hybrid margin certificate failed')
      rows.append(dict(**metadata(r,mode),donor_id=donor['id'],donor_type=donor_type,normalization=policy,attention=int(A),gates=int(G),margin=margin,
       margin_gain=margin-float(C['z'][r['g']]-C['z'][r['c']]),margin_positive=margin>0,gold_top1=int(H['z'].argmax())==r['g'],certificate=error,**{'origin_'+k:v for k,v in origins.items()}))
   del D
 return dict(id=r['id'],success_donors=len(donors['success']),failure_donors=len(donors['failure_control']),rows=rows)


def search_paths(base,cfg):
 return [(mode,'greedy' if seed is None else f'random_{seed}',seed,base/(mode+('_greedy' if seed is None else f'_random_{seed}')+'.json')) for mode in cfg['edit_modes'] for seed in [None]+cfg['random_search_seeds']]


def run_gpu(source,out,cfg):
 spec=source['spec'];records=source['records'];mode=source['mode'];out.mkdir(parents=True,exist_ok=True)
 print(f"Loading {spec['repo']} at {spec['sha']} ({mode})",flush=True)
 if cfg['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
 torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['sha'])
 model=AutoModelForCausalLM.from_pretrained(spec['repo'],revision=spec['sha'],torch_dtype=torch.float32,attn_implementation='eager').to(cfg['device']);engine=Engine(model)
 bases=load_manifest(cfg['subspace_manifest'],spec['sha'],model.config.hidden_size) if cfg['run_subspace'] else None
 frequency,freqmeta=frequency_vector(cfg['frequency_file'] or cfg['frequency_files'].get(spec['key']),engine.head.weight.shape[0],spec['sha'],tokenizer_hash(tok)) if cfg['run_sink'] else (None,{'status':'disabled'})
 dump(out/'external_data_status.json',dict(frequency=freqmeta,gradient_geometry='enabled' if bases is not None else 'not_run: actual pretraining gradient basis not supplied or stage disabled'))
 selected=[r for r in records if r.get('eligible') and r['split']=='evaluation']
 if cfg['max_examples'] is not None:selected=selected[:cfg['max_examples']]
 dump(out/'cohort.json',dict(eligible_evaluation=len(selected),ids=[r['id'] for r in selected],full_cohort=cfg['max_examples'] is None))
 for index,r in enumerate(selected):
  key=digest(r['id'])[:24];meta=metadata(r,mode);hp=out/'hybrids'/f'{key}.json';vp=out/'vectors'/f'{key}.npz';gp=out/'geometry'/f'{key}.json';ep=out/'edit_search'/key;dp=ep/'donor.json'
  want_h=cfg['run_hybrids'] and not r['correct'] and not hp.exists()
  want_v=cfg['run_sink'] and not (vp.exists() and vp.with_suffix('.json').exists())
  want_g=cfg['run_subspace'] and bases is not None and not gp.exists()
  want_e=cfg['run_edit_search'] and not r['correct']
  if want_e and dp.exists():
   donor=json.loads(dp.read_text())
   if not donor['verified'] or all(p.exists() for _,_,_,p in search_paths(ep,cfg)):want_e=False
  if not any([want_h,want_v,want_g,want_e]):continue
  C=engine.capture(torch.tensor(r['input_ids'],device=engine.device));native_margin=float(C['z'][r['g']]-C['z'][r['c']])
  if abs(native_margin-r['margin'])>cfg['certificate_tolerance']*max(1,abs(r['margin'])):raise AssertionError('Native margin differs from input cache: verify exact checkpoint/token IDs/software')
  dump(out/'native_checks'/f'{key}.json',dict(**meta,margin=native_margin,cached_margin=r['margin'],target_rank=int((C['z']>C['z'][r['g']]).sum())+1,certificate=C['certificate']))
  if want_v:vector_snapshot(engine,C,r['source'],vp,meta,cfg)
  if want_h:dump(hp,hybrid_rows(engine,C,r,records,cfg,mode))
  if want_g:
   rows=analyze(engine,C,r['source'],r['g'],r['c'],bases,cfg);dump(gp,dict(rows=[dict(**meta,**row) for row in rows]))
  if want_e:
   # Existing original failures with positive first-token margin are not edit-distance failures.
   if native_margin>0:dump(dp,dict(id=r['id'],verified=False,initial_margin=native_margin,reason='Already positive first-token margin; full-answer failure is a different endpoint'))
   else:
    D,mapping,audit=paraphrase_donor(engine,tok,dict(r,mode=mode),cfg)
    dump(dp,dict(id=r['id'],verified=D is not None,initial_margin=native_margin,audit=audit,mapping=mapping))
    if D is not None:
     targets=aligned_coefficients(C,D,mapping)
     for edit_mode,control,seed,p in search_paths(ep,cfg):
      if p.exists():continue
      print(f"{spec['key']}/{mode} {index+1}/{len(selected)} {r['id']}: {edit_mode} {control}",flush=True)
      result=greedy(engine,C,targets,r['g'],r['c'],budget=cfg['edit_budget'],mode=edit_mode,scope=cfg['edit_scope'],prune=cfg['pruning'] and seed is None,random_seed=seed,tol=cfg['certificate_tolerance'])
      dump(p,dict(result,id=r['id'],group_id=r['group_id'],label=group_label(r),regime=regime(r),mode=edit_mode,control=control))
     del D,targets
  print(f"{spec['key']}/{mode}: {index+1}/{len(selected)} follow-up cases saved",flush=True)
  del C
 if cfg['run_sink']:aggregate_snapshots(out/'vectors',out/'reports',frequency)
 report(out,cfg)
 # Additional matched analysis with actual vocabulary rank, separate from cache-only analysis.
 checks={json.loads(p.read_text())['id']:json.loads(p.read_text()) for p in (out/'native_checks').glob('*.json')}
 enriched=[dict(r,target_rank=checks[r['id']]['target_rank']) if r['id'] in checks else r for r in records]
 if cfg['run_cpu']:posthoc(enriched,out/'rank_adjusted_cpu',cfg)
 del engine,model
 if torch.cuda.is_available():torch.cuda.empty_cache()


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--config');p.add_argument('--cpu-only',action='store_true');p.add_argument('--report-only',action='store_true');a=p.parse_args()
 cfg=dict(DEFAULT);cfg.update(json.loads(Path(a.config).read_text()) if a.config else {});validate(cfg)
 out=Path(a.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'.run.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Another worker holds this results directory')
 input_root,ss=sources(a.input,out/'inputs');protocol=dict(config=cfg,sources=[s['source_hash'] for s in ss],code_hash=code_hash(),external_hashes=external_hashes(cfg))
 pp=out/'protocol.json'
 if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Configuration, source or code changed. Use a new output directory.')
 dump(pp,protocol)
 import transformers,numpy
 dump(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,cuda=torch.version.cuda))
 for source in ss:
  dest=out/source['spec']['key']/source['mode'];dest.mkdir(parents=True,exist_ok=True)
  if a.report_only:
   if (dest/'vectors').exists():
    # Frequency fingerprint was checked during generation. Reload via saved tokenizer only for frequency reports.
    freq_path=cfg['frequency_file'] or cfg['frequency_files'].get(source['spec']['key'])
    if freq_path:
     tok=AutoTokenizer.from_pretrained(source['spec']['repo'],revision=source['spec']['sha'])
     sample=next((dest/'vectors').glob('*.npz'),None)
     if sample:
      import numpy as np
      with np.load(sample,allow_pickle=False) as z:vocab=len(z['subject'])
      f,_=frequency_vector(freq_path,vocab,source['spec']['sha'],tokenizer_hash(tok))
     else:f=None
    else:f=None
    aggregate_snapshots(dest/'vectors',dest/'reports',f)
   report(dest,cfg)
  else:
   if cfg['run_cpu']:
    print('Running cached CPU analyses',flush=True);posthoc(source['records'],dest/'cpu',cfg)
    matched_repairs(source['records'],input_root/'reports/repairs.csv',dest/'cpu',cfg,source['spec']['key'],source['mode'])
   if not a.cpu_only:run_gpu(source,dest,cfg)
 dump(out/('CPU_COMPLETE.json' if a.cpu_only else 'REPORTS_COMPLETE.json' if a.report_only else 'RUN_COMPLETE.json'),dict(status='complete',missing_external_stages=(['pretraining_gradient_geometry'] if not cfg['subspace_manifest'] else [])+(['frequency_alignment'] if not cfg['frequency_file'] and not cfg['frequency_files'] else [])))
 print('Completed:',out,flush=True)

if __name__=='__main__':main()
