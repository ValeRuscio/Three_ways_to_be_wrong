import sys,tempfile,json
from pathlib import Path
from unittest import TestCase,mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from test_core import tiny
from channel_study.io import load
from channel_study.run import run
from channel_study.core import live

class WorkflowTests(TestCase):
 def test_all_six_stages_and_resume(self):
  class Backend:
   def to_str(self):return 'test-tokenizer'
  class Tok:backend_tokenizer=Backend()
  def model_load(spec,device):
   e,_=tiny('olmo2' if 'olmo' in spec['repo'] else 'llama')
   if spec.get('step')==20:
    with torch.no_grad():
     for p in e.model.parameters():p.add_(.001)
   return Tok(),e
  def token(tok,r,device):return torch.tensor([1,4,8,10,5]),[1,2],6
  def gen(e,tok,ids,aliases,n):
   with torch.no_grad():correct=bool(live(e,ids).argmax()==6)
   return dict(generated='accepted' if correct else 'wrong',answer_correct=correct)
  def pin(spec,path):
   from channel_study.io import dump
   s=dict(spec,sha='test');dump(path,s);return s
  refs=dict(final=dict(repo='olmo',sha='test',step=20),earlier=[dict(repo='olmo',sha='old',step=10)])
  cfg=load(Path(__file__).resolve().parents[1]/'config.json')
  cfg.update(device='cpu',models=[dict(key='tiny',repo='tiny')],templates=[0],cpu_threads=2,
   probe_facts=1,probe_layers=[0],families=['gate','down'],frontier_families=['gate','down'],
   seeds=[0],doses=[.001],specificity_n=1,fisher_draws=2,geometry_rank=2,training_rank=2,
   svd_oversample=1,bootstrap=10,training_models=[dict(key='tinyolmo',repo='olmo')],training_facts=2,training_families=['gate','down'])
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);prompts=root/'prompts.jsonl'
   prompts.write_text(''.join(json.dumps(dict(id=str(i),group_id=str(i),relation='rel',template=0,prompt=f'Fact {i}',subject=str(i),answer='accepted',aliases=['accepted'],split='evaluation'))+'\n' for i in range(2)))
   with mock.patch('channel_study.run.model_load',side_effect=model_load),mock.patch('channel_study.run.pin',side_effect=pin),mock.patch('channel_study.run.tokenize',side_effect=token),mock.patch('channel_study.run.generate',side_effect=gen),mock.patch('channel_study.experiments.generate',side_effect=gen),mock.patch('channel_study.training.model_load',side_effect=model_load),mock.patch('channel_study.training.revisions',return_value=refs),mock.patch('channel_study.training.tokenize',side_effect=token),mock.patch('channel_study.training.generate',side_effect=gen):
    stages=['split','fisher','geometry','perturb','frontier','training']
    run(cfg,prompts,root/'out',stages)
    for stage in stages:self.assertTrue((root/'out/reports'/f'{stage}_summary.csv').exists(),stage)
    measurements=list((root/'out/tiny/prompts').glob('*/frontier/*.json'))
    self.assertTrue(measurements)
    times={str(p):p.stat().st_mtime_ns for p in measurements}
    run(cfg,prompts,root/'out',stages)
    self.assertEqual(times,{str(p):p.stat().st_mtime_ns for p in measurements})
