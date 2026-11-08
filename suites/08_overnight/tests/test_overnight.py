import sys,tempfile,json,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from test_followup import tiny
from followup_study.mechanisms import singleton_freezes
from followup_study.cached_cpu import analyze
from followup_study.features import Moments
from channel_study.io import dump

class OvernightTests(unittest.TestCase):
 def test_singletons_baseline_and_endpoint(self):
  for kind in ['llama','qwen2','olmo2']:
   e,C=tiny(kind);r=singleton_freezes(e,C,6,7)
   self.assertLess(r['max_baseline_disagreement'],1e-5)
   end={v['condition']:v for v in r['rows'] if v['layer']==1 and v['family']=='down'}
   self.assertLess(end['attention']['selection_norm'],1e-5)
   self.assertLess(end['gate']['selection_norm'],1e-5)
   self.assertAlmostEqual(end['norm']['selection_norm'],end['all']['selection_norm'],places=5)
 def test_cached_cpu_known_design_and_missing_sink(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);prev=root/'previous';out=root/'cpu';rng=np.random.default_rng(2)
   X=rng.normal(size=(4,3,2,2,5))
   for n in range(4):
    for t in range(3):
     p=prev/'llama3b'/'prompts'/f'{n}_{t}';p.mkdir(parents=True)
     dump(p/'baseline.json',{'prompt':dict(split='evaluation',template=t,group_id=str(n),id=str(n),relation='r')})
     torch.save(torch.from_numpy(X[n,t]),p/'origin_vectors.pt')
     dump(p/'geometry.json',[dict(layer=l,first_norm=2.) for l in range(2)])
   analyze(prev,out,models=['llama3b'])
   import csv
   with (out/'origin_input_anova.csv').open() as f:rows=list(csv.DictReader(f))
   total=((X-X.mean((0,1)))**2).sum((0,1,-1))
   for r in rows:self.assertAlmostEqual(float(r['total_ss']),total[int(r['layer']),int(r['site'])])
   self.assertFalse((out/'sink_vector_constancy.csv').exists())
   state=Moments(prev/'llama3b'/'sink_state'/'relation.npz')
   for n in range(4):state.add(str(n),rng.normal(size=(3,2,3,5)))
   analyze(prev,root/'cpu2',models=['llama3b'])
   self.assertTrue((root/'cpu2'/'sink_vector_constancy.csv').exists())
 def test_singleton_report(self):
  from followup_study.report import report
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cfg=json.loads((Path(__file__).parents[1]/'followup_study/config.json').read_text());cfg.update(models=[{'key':'tiny','repo':'tiny'}],freeze_mode='singletons',bootstrap=5)
   e,C=tiny('llama');p=root/'tiny'/'prompts'/'p'
   dump(p/'baseline.json',dict(prompt=dict(split='evaluation',id='x',group_id='x',relation='r',template=0),answer_correct=False,canonical_top1=False))
   dump(p/'freeze.json',singleton_freezes(e,C,6,7));report(root,cfg)
   self.assertTrue((root/'reports/freeze_summary.csv').exists())
 def test_queue_order_cohort_disk_and_fresh_names(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  from followup_study.overnight import run
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);commands=[]
   class Process:
    pid=123;returncode=0
    def __init__(self,command,**kwargs):commands.append(command)
    def poll(self):return 0
    def wait(self):return 0
   cfg={'models':[{'key':'llama1b','repo':'tiny'}]}
   jobs=[dict(name='nulls_new',models=['llama1b'],stages=['nulls']),dict(name='matrix_new',models=['llama1b'],stages=['matrix'])]
   plan=dict(root=str(root/'campaign'),previous=None,config=cfg,jobs=jobs,minimum_free_gib=20)
   p=root/'plan.json';dump(p,plan);cohort={'evaluation':[{'id':'same'}],'calibration':[]}
   with patch('followup_study.overnight.prepare',return_value=cohort),patch('followup_study.overnight.subprocess.Popen',Process),patch('followup_study.overnight.shutil.disk_usage',side_effect=[SimpleNamespace(free=30*2**30),SimpleNamespace(free=10*2**30)]):
    run(p)
   result=json.loads((root/'campaign/exit.json').read_text())
   self.assertTrue(result['failed']);self.assertEqual(result['jobs'][1]['status'],'skipped_low_disk')
   self.assertEqual(json.loads((root/'campaign/nulls_new/data/cohort.json').read_text()),cohort)
   self.assertEqual(len(commands),2) # CPU and first GPU only
   with self.assertRaises(RuntimeError):run(p)

if __name__=='__main__':unittest.main()
