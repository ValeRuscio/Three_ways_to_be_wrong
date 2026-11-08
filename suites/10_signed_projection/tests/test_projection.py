import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,Olmo2Config,Olmo2ForCausalLM
from causal_suite.engine import Engine
from followup_study.mechanisms import capture,collect,MECHANISMS
from projection_study.core import measure,projections,finite_checks,CONDITIONS
from projection_study.helpers import reference_rows
from projection_study.run import run
from channel_study.io import dump,load

def tiny(kind='llama'):
 torch.manual_seed(2);C,M={'llama':(LlamaConfig,LlamaForCausalLM),'qwen':(Qwen2Config,Qwen2ForCausalLM),'olmo':(Olmo2Config,Olmo2ForCausalLM)}[kind]
 cfg=C(vocab_size=31,hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.);cfg._attn_implementation='eager'
 e=Engine(M(cfg).float());return e,capture(e,torch.tensor([1,4,8,10,5]),1e-4)

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2)
 def test_baselines_and_signed_projection(self):
  for kind in ['llama','qwen','olmo']:
   e,C=tiny(kind);mean=torch.randn(2,16);d,G=measure(e,C,6,7,mean)
   self.assertLess(d['baseline_error'],1e-5)
   for row in d['rows']:
    l=row['layer'];u=mean[l].double()/mean[l].double().norm();actual=float(G[row['condition']][l].double()@u)
    self.assertAlmostEqual(row['projection'],actual,places=10)
   for l in range(2):
    rows={r['condition']:r for r in d['rows'] if r['layer']==l}
    self.assertAlmostEqual(d['paired'][l]['norm_minus_gate'],rows['scales_only']['removed_projection']-rows['gates_only']['removed_projection'],places=10)
 def test_zero_mean_undefined_and_common_scale(self):
  G={k:torch.ones(2,3)*(i+1) for i,k in enumerate(CONDITIONS)};mean=torch.zeros(2,3);rows,_=projections(G,mean)
  self.assertTrue(all(r['projection'] is None for r in rows))
  mean[0]=1;rows,_=projections(G,mean);d={r['condition']:r for r in rows if r['layer']==0}
  for k in CONDITIONS:self.assertAlmostEqual(d[k]['removed_over_content_norm'],d['live']['projection_over_content_norm']-d[k]['projection_over_content_norm'])
 def test_directional_certificates_all_conditions(self):
  for kind in ['llama','qwen','olmo']:
   e,C=tiny(kind);mean=torch.randn(2,16);_,G=measure(e,C,6,7,mean)
   checks=finite_checks(e,C,6,7,G['live'],G,dict(fd_layer_fractions=[0,1],fd_epsilons=[1e-3],fd_atol=1e-6,fd_rtol=.03))
   self.assertEqual(len(checks),10);self.assertTrue(all(c['passed'] for c in checks));self.assertGreaterEqual(sum(c['resolved'] for c in checks),4)
 def test_reference_subject_separation(self):
  cohort=dict(evaluation=[dict(group_id='1')],calibration=[dict(group_id=str(i),id=str(i),template=0,prompt=str(i)) for i in range(5)])
  selected=reference_rows(cohort,4,0);self.assertNotIn('1',{r['group_id'] for r in selected})
 def test_pipeline_report_and_explicit_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);evidence=root/'evidence';out=root/'out'
   def row(i,split,t=0):return dict(id=str(i),group_id=str(i),relation='r',template=t,split=split,prompt=f'{i}-{t}',subject=str(i),answer='gold',aliases=['gold'])
   dump(evidence/'data/cohort.json',dict(evaluation=[row(i,'evaluation') for i in range(5)],calibration=[row(10+i,'calibration') for i in range(5)]));dump(evidence/'llama1b/model.json',dict(repo='tiny',sha='test'))
   cfg=load(Path(__file__).resolve().parents[1]/'projection_study/config.json');cfg.update(models=['llama1b'],device='cpu',cpu_threads=2,prompts=5,reference_subjects=5,fd_prompts=1,fd_epsilons=[1e-4],bootstrap=5)
   def tok(*a):return torch.tensor([1,4,8,10,5]),[1,2],6
   with patch('projection_study.run.model_load',side_effect=lambda *a:(object(),tiny()[0])) as loader,patch('projection_study.run.tokenize',side_effect=tok):
    run(cfg,out,evidence);paths=list(out.glob('llama1b/prompts/*/measurement.json'));self.assertEqual(len(paths),5)
    times=[p.stat().st_mtime_ns for p in paths];run(cfg,out,evidence);self.assertEqual(times,[p.stat().st_mtime_ns for p in paths]);self.assertEqual(loader.call_count,1)
   for file in ['projections_summary.csv','paired_contrasts_summary.csv','finite_difference_summary.csv','audit.json','llama1b_signed_projection.pdf']:self.assertTrue((out/'reports'/file).exists())
   self.assertEqual(load(out/'reports/audit.json')['finite_failures'],0)
   self.assertEqual(len(list(out.glob('llama1b/prompts/*/gradients.npz'))),5)
   cfg['seed']=1
   with self.assertRaises(RuntimeError):run(cfg,out,evidence)

if __name__=='__main__':unittest.main()
