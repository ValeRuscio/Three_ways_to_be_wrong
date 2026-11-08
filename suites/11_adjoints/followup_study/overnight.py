"""Detached sequential GPU queue plus independent read-only CPU analysis."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from channel_study.io import load,dump
from .data import prepare


def run(plan_path):
 plan=load(plan_path);root=Path(plan['root']);root.mkdir(parents=True,exist_ok=True)
 with (root/'queue.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (root/'started.json').exists():raise RuntimeError('Campaign already launched. Use a fresh campaign; no automatic resume.')
  dump(root/'started.json',dict(pid=os.getpid(),time=time.time(),plan=plan))
  cpu_out=root/'00_cached_cpu';cpu_log=(root/'cpu.log').open('a',buffering=1)
  cmd=[sys.executable,'-u','-m','followup_study.cached_cpu','--out',str(cpu_out)]
  if plan['previous']:cmd+=['--previous',plan['previous']]
  cpu=subprocess.Popen(cmd,stdout=cpu_log,stderr=subprocess.STDOUT)
  outcome=[];failed=False;pins={}
  try:
   # Prepare ONCE, then copy identical cohort into every new job directory.
   cohort_root=root/'cohort_source';cohort=prepare(plan['config'],cohort_root,plan['previous'])
   for job in plan['jobs']:
    if (root/'STOP').exists():break
    jobroot=root/job['name']
    if jobroot.exists():raise RuntimeError('Job directory already exists: '+str(jobroot))
    free=shutil.disk_usage(root).free/2**30
    if free<plan['minimum_free_gib']:
     outcome.append(dict(job=job['name'],status='skipped_low_disk',free_gib=free));failed=True
     print('Stopping queue: insufficient free disk',free,flush=True);break
    cfg=dict(plan['config']);cfg['models']=[m for m in cfg['models'] if m['key'] in job['models']]
    dump(jobroot/'data/cohort.json',cohort);dump(jobroot/'configuration.json',cfg)
    for m in cfg['models']:
     if m['key'] in pins:dump(jobroot/m['key']/'model.json',pins[m['key']])
    command=[sys.executable,'-u','-m','followup_study.run','--config',str(jobroot/'configuration.json'),'--out',str(jobroot),'--stages',*job['stages']]
    if plan['previous']:command+=['--previous',plan['previous']]
    print('START',job['name'],'free GiB',round(free,1),flush=True)
    with (jobroot/'execution.log').open('a',buffering=1) as log:
     worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
     dump(root/'active.json',dict(job=job['name'],pid=worker.pid,log=str(jobroot/'execution.log')))
     while worker.poll() is None:
      if (root/'STOP').exists():(jobroot/'STOP').touch()
      time.sleep(2)
    stopped=(jobroot/'STOP').exists();rc=worker.returncode
    for m in cfg['models']:
     pinpath=jobroot/m['key']/'model.json'
     if pinpath.exists():pins[m['key']]=load(pinpath)
    result=dict(job=job['name'],returncode=rc,stopped=stopped,finished=time.time())
    dump(jobroot/'exit.json',result);outcome.append(result);dump(root/'progress.json',outcome)
    print('END',result,flush=True)
    if rc or stopped:failed=bool(rc);break
  except BaseException as ex:
   failed=True;dump(root/'error.json',dict(type=type(ex).__name__,message=str(ex)));raise
  finally:
   # CPU normally finishes far earlier; keep its status visible independently.
   cpu_rc=cpu.wait();cpu_log.close()
   dump(root/'exit.json',dict(failed=failed or cpu_rc!=0,stopped=(root/'STOP').exists(),cpu_returncode=cpu_rc,jobs=outcome,finished=time.time()))


def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();run(a.plan)
if __name__=='__main__':main()
