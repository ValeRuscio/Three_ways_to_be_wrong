import argparse,fcntl,os,shutil,subprocess,sys,time
from pathlib import Path
from channel_study.io import load,dump


def run(plan_path):
 plan=load(plan_path);root=Path(plan['root']);root.mkdir(parents=True,exist_ok=True)
 with (root/'queue.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (root/'started.json').exists():raise RuntimeError('Campaign already launched; create a new campaign explicitly.')
  dump(root/'started.json',dict(pid=os.getpid(),started=time.time(),plan=plan))
  evidence=Path(plan['evidence']);cohort=load(evidence/'data/cohort.json');results=[];failed=False
  cpu_log=(root/'cpu.log').open('a',buffering=1)
  command=[sys.executable,'-u','-m','precision_study.recalibrate','--evidence',str(evidence),'--out',str(root/'00_recalibration')]
  if plan.get('raw_profiles'):command+=['--raw-profiles',*plan['raw_profiles']]
  cpu=subprocess.Popen(command,stdout=cpu_log,stderr=subprocess.STDOUT)
  try:
   for job in plan['jobs']:
    if (root/'STOP').exists():break
    free=shutil.disk_usage(root).free/2**30
    if free<plan['minimum_free_gib']:raise RuntimeError(f'Free disk {free:.1f} GiB below floor')
    out=root/job['name']
    if out.exists():raise RuntimeError('New output directory required: '+str(out))
    cfg=dict(plan['config']);cfg['models']=[m for m in cfg['models'] if m['key'] in job['models']]
    dump(out/'configuration.json',cfg);dump(out/'data/cohort.json',cohort)
    for model in job['models']:dump(out/model/'model.json',load(evidence/model/'model.json'))
    if job['stage']=='anisotropy':
     command=[sys.executable,'-u','-m','precision_study.anisotropy','--config',str(out/'configuration.json'),'--out',str(out),'--evidence',str(evidence),'--stop-root',str(root)]
    else:
     command=[sys.executable,'-u','-m','followup_study.run','--config',str(out/'configuration.json'),'--out',str(out),'--stages',job['stage']]
    print('START',job['name'],flush=True)
    with (out/'execution.log').open('a',buffering=1) as log:
     worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
     dump(root/'active.json',dict(job=job['name'],pid=worker.pid,log=str(out/'execution.log')))
     while worker.poll() is None:
      if (root/'STOP').exists():(out/'STOP').touch()
      time.sleep(2)
     rc=worker.returncode
     if job['stage']=='certificate' and rc==0 and not (out/'STOP').exists():
      rc=subprocess.run([sys.executable,'-u','-m','precision_study.certificate_report','--root',str(out)],stdout=log,stderr=subprocess.STDOUT).returncode
    result=dict(job=job['name'],returncode=rc,stopped=(out/'STOP').exists() or (root/'STOP').exists(),finished=time.time())
    dump(out/'exit.json',result);results.append(result);dump(root/'progress.json',results);print('END',result,flush=True)
    if rc or result['stopped']:failed=bool(rc);break
  except BaseException as ex:
   failed=True;dump(root/'error.json',dict(type=type(ex).__name__,message=str(ex)));raise
  finally:
   cpu_rc=cpu.wait();cpu_log.close()
   dump(root/'exit.json',dict(failed=failed or cpu_rc!=0,stopped=(root/'STOP').exists(),cpu_returncode=cpu_rc,jobs=results,finished=time.time()))


def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();run(a.plan)
if __name__=='__main__':main()
