import argparse,fcntl,os,subprocess,sys,time
from pathlib import Path
from channel_study.io import dump


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--evidence',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
 with (root/'supervisor.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (root/'launch.json').exists() and not a.resume:raise RuntimeError('Already launched; monitor, or explicitly resume with identical protocol.')
  if (root/'STOP').exists():raise RuntimeError('STOP exists; clear it only after prior worker exit.')
  (root/'exit.json').unlink(missing_ok=True)
  command=[sys.executable,'-u','-m','adjoint_study.run','--config',a.config,'--out',a.out,'--evidence',a.evidence]
  with (root/'execution.log').open('a',buffering=1) as log:
   worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
   dump(root/'launch.json',dict(supervisor_pid=os.getpid(),worker_pid=worker.pid,command=command,started=time.time()))
   rc=worker.wait()
  dump(root/'exit.json',dict(returncode=rc,stopped=(root/'STOP').exists(),finished=time.time()))
if __name__=='__main__':main()
