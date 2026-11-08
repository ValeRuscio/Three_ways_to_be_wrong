import sys,json,tempfile
from pathlib import Path
from unittest import TestCase,mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from test_followup import tiny
from followup_study.run import run
from channel_study.io import load,dump

class WorkflowTests(TestCase):
 def test_all_eight_experiments_and_resume(self):
  cfg=load(Path(__file__).resolve().parents[1]/'followup_study/config.json')
  cfg.update(device='cpu',cpu_threads=2,models=[dict(key='llama1b',repo='tiny')],templates=[0,1,2],calibration_templates=[0],
   null_prompts=1,freeze_prompts=1,matrix_prompts=1,certificate_prompts=1,readout_seeds=[0],seeds=[0],fd_epsilons=[1e-4],bootstrap=10,min_calibration_subjects=2)
  def model_load(spec,device):return object(),tiny('llama')[0]
  def pin(spec,path):s=dict(spec,sha='test');dump(path,s);return s
  def tokenize(tok,r,device):return torch.tensor([1,4,8,10,5]),[1,2],6
  def generate(e,tok,ids,aliases,n):return dict(generated='wrong',answer_correct=False)
  ev=[dict(id=str(i),group_id=str(i),relation='director',split='evaluation',template=t,prompt=f'Fact {i} template {t}',subject=str(i),answer='gold',aliases=['gold']) for i in range(2) for t in range(3)]
  cal=[dict(ev[0],id='cal'+str(i),group_id='cal'+str(i),split='calibration',template=0) for i in range(2)]
  cohort=dict(evaluation=ev,calibration=cal,dataset_revision='test',previous=None)
  stages=['profiles','regimes','variance','sink','nulls','freeze','certificate','matrix']
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   with mock.patch('followup_study.run.prepare',return_value=cohort),mock.patch('followup_study.run.pin',side_effect=pin),mock.patch('followup_study.run.model_load',side_effect=model_load),mock.patch('followup_study.run.tokenize',side_effect=tokenize),mock.patch('followup_study.run.generate',side_effect=generate),mock.patch('transformers.AutoConfig.from_pretrained',side_effect=lambda *a,**k:tiny('llama')[0].model.config),mock.patch('transformers.AutoTokenizer.from_pretrained',return_value=object()):
    run(cfg,root,stages)
    for name in ['profile_summary.csv','regimes.csv','variance_summary.csv','sink_summary.csv','null_paired_summary.csv','freeze_summary.csv','certificate_summary.csv','matrix_summary.csv']:
     self.assertTrue((root/'reports'/name).exists(),name)
    files=list((root/'llama1b/prompts').glob('*/freeze.json'));times={str(p):p.stat().st_mtime_ns for p in files}
    run(cfg,root,stages)
    self.assertEqual(times,{str(p):p.stat().st_mtime_ns for p in files})
