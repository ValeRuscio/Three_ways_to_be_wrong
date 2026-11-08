"""Run each suite in a fresh process to avoid importing the wrong engine version."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',action='append');p.add_argument('--report',type=Path);a=p.parse_args()
    results=[]
    for suite in sorted((ROOT/'suites').iterdir()):
        if a.suite and suite.name not in a.suite:continue
        test=suite/'tests'
        if not test.exists():test=suite/'revision/tests'
        if not test.exists():continue
        print('TEST',suite.name,flush=True)
        env=dict(os.environ,PYTHONPATH=str(suite),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        start=time.time()
        r=subprocess.run([sys.executable,'-m','pytest','-q',str(test)],cwd=suite,env=env)
        results.append(dict(suite=suite.name,returncode=r.returncode,seconds=round(time.time()-start,2)))
    if not results:raise SystemExit('No tests selected')
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(results,indent=2)+'\n')
    raise SystemExit(1 if any(r['returncode'] for r in results) else 0)
if __name__=='__main__':main()
