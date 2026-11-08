import sys,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch,numpy as np
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,Olmo2Config,Olmo2ForCausalLM
from causal_suite.engine import Engine
from channel_study.core import factors,modules,dot
from followup_study.mechanisms import capture,collect,MECHANISMS,three_way,certificates,layer_matrix,subsets,freezes
from followup_study.features import Crossed,Moments,feature_vectors,origin_vectors
from followup_study.regimes import calibrate,classify

torch.set_num_threads(2)

def tiny(kind):
 torch.manual_seed(12)
 c=dict(vocab_size=31,hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.,pad_token_id=0,bos_token_id=1,eos_token_id=2)
 C,M={'llama':(LlamaConfig,LlamaForCausalLM),'qwen2':(Qwen2Config,Qwen2ForCausalLM),'olmo2':(Olmo2Config,Olmo2ForCausalLM)}[kind]
 cfg=C(**c);cfg._attn_implementation='eager';e=Engine(M(cfg).float());ids=torch.tensor([1,4,8,10,5]);return e,capture(e,ids,1e-4)

class FollowupTests(unittest.TestCase):
 def test_native_freeze_matches_affine_and_all_baselines(self):
  for kind in ['llama','qwen2','olmo2']:
   e,C=tiny(kind);obj=lambda z:z[6]-z[7]
   F,fv=factors(e,C,obj,'frozen');H,hv=collect(e,C,obj,MECHANISMS)
   self.assertAlmostEqual(fv,hv,places=5)
   for name,m in modules(e).items():
    a=F[name][0].T@F[name][1] if name in F else torch.zeros_like(m.weight)
    b=H[name][0].T@H[name][1] if name in H else torch.zeros_like(m.weight)
    torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4,msg=kind+'/'+name)
   for mask in subsets():
    _,value=collect(e,C,obj,mask);self.assertAlmostEqual(value,float(obj(C['z'])),places=5)

 def test_shapley_gradient_closure_and_normalization_endpoint(self):
  for kind in ['llama','qwen2','olmo2']:
   e,C=tiny(kind);d=three_way(e,C,6,7)
   for row in d['rows']:
    self.assertLess(row['closure'],1e-4)
    shares=[row[k+'_signed_projection_share'] for k in MECHANISMS]
    if all(v is not None for v in shares):self.assertAlmostEqual(sum(shares),1,places=4)
   # Last down projection: only normalization can reselect downstream.
   row=next(r for r in d['rows'] if r['layer']==1 and r['family']=='down')
   self.assertLess(row['attention_norm'],1e-5);self.assertLess(row['gate_norm'],1e-5)
   self.assertAlmostEqual(row['norm_signed_projection_share'],1,places=4)

 def test_source_layer_matrix_future_weights_do_not_affect_past_gate(self):
  e,C=tiny('llama');r=layer_matrix(e,C,6,7,{'matrix_families':['up','down','q','k','v','o','gate']})
  for row in r['rows']:
   if row['source_layer']>row['frozen_gate_layer']:self.assertLess(row['effect_norm'],1e-5)

 def test_certificates(self):
  e,C=tiny('qwen2');cfg=dict(certificate_layers_resolved=[0,1],seeds=[0],fd_epsilons=[1e-3,1e-4],fd_atol=2e-4,fd_rtol=.03,frozen_rtol=.002)
  result=certificates(e,C,6,7,cfg)
  self.assertTrue(all(r['live_pass'] and r['frozen_pass'] for r in result['rows']))
  self.assertTrue(any(r['direction']=='gradient_aligned' for r in result['rows']))

 def test_origin_centering_and_role_partition(self):
  for kind in ['llama','qwen2','olmo2']:
   e,C=tiny(kind);o,vec=origin_vectors(e,C,[1,2],6,7,1e-4);f=feature_vectors(e,C,[1,2])
   self.assertLess(o['certificate'],1e-4);self.assertEqual(vec.shape,(2,3,24))
   np.testing.assert_allclose(f['attention'].sum(-1),1,atol=1e-6)
   self.assertAlmostEqual(sum(o['margin_origins'].values()),float(C['z'][6]-C['z'][7]),places=5)
   # Common output bias shifts cancel from every vocabulary-centered diagnostic.
   e.head.bias=torch.nn.Parameter(torch.full((31,),5.),requires_grad=False)
   D=capture(e,C['ids'],1e-4);o2,_=origin_vectors(e,D,[1,2],6,7,1e-4)
   for k in o['centered_gold']:self.assertAlmostEqual(o['centered_gold'][k],o2['centered_gold'][k],places=5)

 def test_streamed_variance_and_sink_moments(self):
  rng=np.random.default_rng(2);X=rng.normal(size=(5,3,2,2,7));Y=rng.normal(size=(5,3,2,3,7))
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);cross=Crossed(path/'cross.npz');sink=Moments(path/'sink.npz')
   for i in range(5):cross.add(str(i),dict(gate=X[i],attention=X[i]));sink.add(str(i),Y[i])
   grand=X.mean((0,1));sub=3*((X.mean(1)-grand)**2).sum((0,-1));tem=5*((X.mean(0)-grand)**2).sum((0,-1));tot=((X-grand)**2).sum((0,1,-1))
   for r in cross.rows():
    l,j=r['layer'],r['site'];self.assertAlmostEqual(r['subject_ss'],sub[l,j]);self.assertAlmostEqual(r['template_ss'],tem[l,j]);self.assertAlmostEqual(r['total_ss'],tot[l,j])
   for r in sink.rows():
    t,l=r['template_slot'],r['layer'];j=['subject','template','first'].index(r['origin']);V=Y[:,t,l,j];v=((V-V.mean(0))**2).sum(-1).mean()
    self.assertAlmostEqual(r['trace_variance'],v)
   resumed=Crossed(path/'cross.npz');resumed.add('0',dict(gate=X[0],attention=X[0]));self.assertEqual(len(resumed.done),5)

 def test_diagnostic_branches_not_forced_to_three(self):
  def record(g,c,gm,tm,answer=False,top1=False):
   return dict(prompt={'relation':'r'},answer_correct=answer,canonical_top1=top1,origins=dict(centered_gold={'subject':g},centered_competitor={'subject':c},gold_source_share=g/10,competitor_source_share=c/10,margin_origins={'subject':gm,'template':tm}))
  refs={'r':{'tau':1,'share':.1}}
  for rec,label in [(record(0,0,0,-1),'absent_origin'),(record(0,3,-3,1),'misretrieved_origin'),(record(3,0,3,-4),'overridden_origin'),(record(3,3,0,-1),'mixed_origin_failure'),(record(3,0,3,-4,top1=True),'continuation_failure'),(record(0,0,0,-1,answer=True),'answer_success')]:self.assertEqual(classify(rec,refs),label)

if __name__=='__main__':unittest.main()
