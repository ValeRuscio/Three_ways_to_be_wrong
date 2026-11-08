import argparse,os,subprocess,sys,time
from pathlib import Path
from channel_study.io import dump

def worker(args):
 a=args
 root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
 command=[sys.executable,'-u','-m','followup_study.run','--config',a.config,'--out',a.out,'--stages',*a.stages]
 if a.previous:command+=['--previous',a.previous]
 with (root/'execution.log').open('a',buffering=1) as log:
  worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
  dump(root/'launch.json',dict(worker_pid=worker.pid,supervisor_pid=os.getpid(),command=command,started=time.time()))
  result=worker.wait()
 dump(root/'exit.json',dict(returncode=result,stopped=(root/'STOP').exists(),finished=time.time()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--previous');p.add_argument('--stages',nargs='+',required=True);a=p.parse_args()
 import fcntl
 root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
 with (root/'supervisor.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Supervisor already running for this output directory')
  worker(a)
if __name__=='__main__':main()
