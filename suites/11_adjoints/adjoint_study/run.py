import argparse,csv,fcntl,gc,json,signal,time,shutil
from pathlib import Path
import torch
from channel_study.io import load,dump,key,digest,model_load,tokenize,environment
from followup_study.data import selected
from followup_study.mechanisms import capture
from .core import measure_adjoints,embedding_checks,gate_jacobian,sgd_experiment,curvature


def regime_map(path):
    out={}
    for r in csv.DictReader(Path(path).read_text().splitlines()):
        k=(r['model'],str(r['id']),str(r['template']))
        if k in out and out[k]!=r['regime']:raise ValueError('Conflicting regime keys; select one threshold first')
        out[k]=r['regime']
    return out


def main_run(cfg,out,evidence):
    out=Path(out);evidence=Path(evidence);stop=out/'STOP'
    if stop.exists():raise RuntimeError('STOP exists. Clear only after previous worker exits.')
    if set(cfg['stages'])-{'adjoints','jacobian','sgd','curvature'}:raise ValueError('Unknown stage')
    torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cohort=load(evidence/'data/cohort.json');rows=cohort['evaluation'];pins={m:load(evidence/m/'model.json') for m in cfg['models']}
    labelpath=Path(cfg['regimes_csv']) if cfg.get('regimes_csv') else evidence/'regimes.csv'
    labels=regime_map(labelpath)
    sources={str(p.relative_to(Path(__file__).parents[1])):p.read_text() for pkg in ['adjoint_study','channel_study','causal_suite','followup_study'] for p in (Path(__file__).parents[1]/pkg).glob('*.py')}
    protocol=dict(config=cfg,pins=pins,cohort_hash=digest(cohort),regimes_hash=digest(labelpath.read_text()),source_hash=digest(sources))
    if (out/'protocol.json').exists() and load(out/'protocol.json')!=protocol:raise RuntimeError('Changed config/code/data; use a new RUN_NAME')
    dump(out/'protocol.json',protocol);dump(out/'environment.json',environment());dump(out/'data/cohort.json',cohort)
    for name,src in sources.items():
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(src)
    (out/'regimes.csv').write_text(labelpath.read_text())
    # Stage-first order guarantees fast adjoints finish for both models before optional jobs.
    for stage in cfg['stages']:
        for model in cfg['models']:
            if stage!='adjoints' and model not in cfg['optional_models']:continue
            keys=selected(rows,cfg[stage+'_prompts'],cfg['seed']+7)
            evaluation=sorted([r for r in rows if key(r) in keys],key=key)
            fdkeys=selected(evaluation,cfg['fd_prompts'],cfg['seed']+19)
            root=out/model/stage;root.mkdir(parents=True,exist_ok=True)
            dump(out/model/'model.json',pins[model])
            if all((root/(key(r)+'.json')).exists() for r in evaluation):continue
            if stop.exists():return
            print(f'Loading {model} for {stage}: {len(evaluation)} prompts',flush=True)
            tok,e=model_load(pins[model],cfg['device'])
            local=dict(cfg,weight_layers=sorted({round(f*(e.L-1)) for f in cfg['weight_layer_fractions']}))
            for i,r in enumerate(evaluation):
                if stop.exists():return
                if shutil.disk_usage(out).free<cfg['min_free_gb']*2**30:raise RuntimeError('Low disk. Completed prompts saved; resume after freeing space.')
                path=root/(key(r)+'.json')
                if path.exists():continue
                started=time.time();ids,S,g=tokenize(tok,r,e.device)
                if len(ids)>cfg['max_tokens']:raise ValueError(f'Prompt exceeds max_tokens: {key(r)}')
                C=capture(e,ids,cfg['tolerance']);z=C['z'].clone();z[g]=-torch.inf;c=int(z.argmax())
                reg=labels.get((model,str(r['id']),str(r['template'])),'unlabelled')
                result=dict(model=model,prompt_key=key(r),id=str(r['id']),group_id=str(r['group_id']),relation=r['relation'],template=r['template'],
                    regime=reg,g=g,c=c,baseline_margin=float(C['z'][g]-C['z'][c]),native_certificate=C['certificate'],tokens=len(ids),subject_positions=S)
                if stage=='adjoints':
                    data,L,F=measure_adjoints(e,C,g,c,S);data['finite_checks']=embedding_checks(e,C,g,c,S,L,F,cfg) if key(r) in fdkeys else []
                    del L,F
                elif stage=='jacobian':data=gate_jacobian(e,C,S,cfg,stop)
                elif stage=='sgd':data=sgd_experiment(e,C,g,c,local,stop)
                else:data=curvature(e,C,g,c,local,stop)
                if data is None:return
                result.update(data=data,seconds=time.time()-started);dump(path,result);del C;gc.collect()
                print(f'{model}/{stage}: {i+1}/{len(evaluation)} saved ({result["seconds"]:.1f}s), regime={reg}',flush=True)
            del e,tok;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            from .report import report
            report(out,cfg)
    from .report import report
    report(out,cfg);dump(out/'completion.json',dict(finished=time.time(),stages=cfg['stages'],models=cfg['models']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--evidence',required=True);p.add_argument('--report-only',action='store_true');a=p.parse_args()
    cfg=load(a.config);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.report_only:
        from .report import report
        report(out,cfg);return
    with (out/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:(out/'STOP').touch())
        main_run(cfg,out,a.evidence)
if __name__=='__main__':main()
