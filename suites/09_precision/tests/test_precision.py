import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from test_followup import tiny
from precision_study.anisotropy import residual_gradients,measure,reference_rows,output_projection
from precision_study.recalibrate import folds,references,label
from precision_study.certificate_report import report
from followup_study.mechanisms import MECHANISMS,certificates
from channel_study.io import dump,load

class PrecisionTests(unittest.TestCase):
 def test_residual_gradient_finite_difference(self):
  e,C=tiny('llama');L,lv=residual_gradients(e,C,6,7);F,fv=residual_gradients(e,C,6,7,MECHANISMS)
  torch.manual_seed(7);v=torch.randn(16);v/=v.norm();eps=1e-3
  from followup_study.mechanisms import freezes
  for which,G in [((),L),(MECHANISMS,F)]:
   vals=[]
   for sign in [-1,1]:
    def edit(m,args,kw):
     x=(args[0] if args else kw['hidden_states']).clone();x[0,-1]+=sign*eps*v
     if args:return (x,*args[1:]),kw
     return args,dict(kw,hidden_states=x)
    with freezes(e,C,which):
     h=e.blocks[1].register_forward_pre_hook(edit,with_kwargs=True)
     try:
      with torch.no_grad():z=e.model(C['ids'][None],use_cache=False).logits[0,-1];vals.append(float(z[6]-z[7]))
     finally:h.remove()
   self.assertAlmostEqual((vals[1]-vals[0])/(2*eps),float(G[1]@v),delta=2e-4)
  rows=measure(e,C,6,7,torch.ones(2,16));self.assertTrue(all(r['closure_error']<1e-10 for r in rows))

 def test_weight_projection_matches_dense(self):
  torch.manual_seed(3);d=torch.randn(5,16);x=torch.randn(5,24);mu=torch.randn(16);G=d.double().T@x.double();u=mu.double()/mu.double().norm()
  self.assertAlmostEqual(output_projection((d,x),mu),float((u@G).norm()/G.norm()),places=10)

 def test_subject_folds_and_reference_disjointness(self):
  mapping=folds([str(i) for i in range(50)]);self.assertEqual(mapping,folds(list(reversed([str(i) for i in range(50)]))))
  rows=[]
  for i in range(50):
   for t in range(3):rows.append(dict(group_id=str(i),relation='r',answer_correct=True,canonical_top1=True,gold_subject=i+1.,gold_source_share=(i+1)/100,fold=mapping[str(i)]))
  d=pd.DataFrame(rows);train=d[d.fold!=0];test=d[d.fold==0];self.assertFalse(set(train.group_id)&set(test.group_id))
  a=references(train,20,.05);d.loc[d.fold==0,'gold_subject']=100000
  self.assertEqual(a,references(d[d.fold!=0],20,.05));self.assertEqual(a['r']['subjects'],40)
  self.assertIsNone(references(train,41,.05)['r']['tau'])
  cohort=dict(evaluation=[dict(group_id='0')],calibration=[dict(group_id=str(i),id=str(i),template=0,prompt=str(i)) for i in range(5)])
  self.assertNotIn('0',{r['group_id'] for r in reference_rows(cohort,4,0)})

 def test_certificate_export_all_failures_visible(self):
  e,C=tiny('llama');cfg=dict(certificate_layers_resolved=[0,1],seeds=[0],fd_epsilons=[1e-4],fd_atol=2e-4,fd_rtol=.03,frozen_rtol=.002)
  result=certificates(e,C,6,7,cfg)
  self.assertEqual(len(result['split_identity']),14)
  self.assertTrue(all('frozen_resolved' in r for r in result['rows']))
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);p=out/'tiny/prompts/key';dump(p/'certificate.json',result);dump(p/'baseline.json',dict(prompt=dict(id='1',group_id='1',template=0)))
   report(out);audit=load(out/'reports/certificate_audit.json');self.assertEqual(audit['checks'],len(result['rows']));self.assertEqual(audit['frozen_failures'],0)

 def test_anisotropy_pipeline(self):
  from unittest.mock import patch
  from precision_study.anisotropy import run
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);evidence=root/'evidence';out=root/'run'
   def row(i,split):return dict(id=str(i),group_id=str(i),relation='r',template=0,prompt=str(i),subject=str(i),answer='gold',aliases=['gold'],split=split)
   dump(evidence/'data/cohort.json',dict(evaluation=[row(1,'evaluation'),row(2,'evaluation')],calibration=[row(3,'calibration'),row(4,'calibration')]))
   dump(evidence/'llama1b/model.json',dict(repo='tiny',sha='test'))
   cfg=dict(seed=0,cpu_threads=2,device='cpu',tolerance=1e-4,anisotropy_reference_subjects=2,anisotropy_prompts=2,anisotropy_init_seeds=[1729],bootstrap=5,models=[dict(key='llama1b',repo='tiny')])
   with patch('precision_study.anisotropy.model_load',side_effect=lambda *a:(object(),tiny('llama')[0])),patch('precision_study.anisotropy.tokenize',side_effect=lambda *a:(torch.tensor([1,4,8,10,5]),[1,2],6)),patch('transformers.AutoConfig.from_pretrained',side_effect=lambda *a,**kw:tiny('llama')[0].model.config),patch('transformers.AutoTokenizer.from_pretrained',return_value=object()):
    run(cfg,out,evidence)
   self.assertTrue((out/'completion.json').exists())
   self.assertTrue((out/'reports/weight_output_projection_summary.csv').exists())
   df=pd.read_csv(out/'reports/residual_anisotropy_per_prompt.csv');self.assertEqual(len(df),8)
   for prompt,d in df.groupby('id'):self.assertEqual(d.c.nunique(),1)

if __name__=='__main__':unittest.main()
