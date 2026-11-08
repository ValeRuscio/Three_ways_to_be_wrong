"""Detached subprocess supervisor. No shell interpolation or token handling."""
import argparse,subprocess,sys,time,os
from pathlib import Path
from .io import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--config',required=True);p.add_argument('--prompts',required=True);p.add_argument('--stages',nargs='+',required=True);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 args=[sys.executable,'-u','-m','channel_study.run','--config',a.config,'--prompts',a.prompts,'--out',a.out,'--stages',*a.stages]
 dump(out/'supervisor.json',dict(pid=os.getpid(),command=args,started=time.time()))
 with (out/'execution.log').open('a',buffering=1) as log:
  worker=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
  dump(out/'launch.json',dict(worker_pid=worker.pid,supervisor_pid=os.getpid(),started=time.time(),command=args))
  result=worker.wait()
 dump(out/'exit.json',dict(returncode=result,stopped=(out/'STOP').exists(),finished=time.time()))
if __name__=='__main__':main()
