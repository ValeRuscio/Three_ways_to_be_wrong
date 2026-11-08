import unittest,sys,tempfile,json,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,Olmo2Config,Olmo2ForCausalLM
from causal_suite.engine import Engine
from channel_study.core import *
from channel_study.experiments import split,fisher,angles
from channel_study.io import import_prompts

torch.set_num_threads(2)

def tiny(kind):
 torch.manual_seed(9)
 conf=dict(vocab_size=31,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
  num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
  attention_dropout=0.,pad_token_id=0,bos_token_id=1,eos_token_id=2)
 klass,model={'llama':(LlamaConfig,LlamaForCausalLM),'qwen2':(Qwen2Config,Qwen2ForCausalLM),'olmo2':(Olmo2Config,Olmo2ForCausalLM)}[kind]
 c=klass(**conf);c._attn_implementation='eager';e=Engine(model(c).float());ids=torch.tensor([1,4,8,10,5])
 return e,e.capture(ids)


class CoreTests(unittest.TestCase):
 def test_factor_gradients_all_architectures_and_families(self):
  for kind in ['llama','qwen2','olmo2']:
   e,C=tiny(kind);rows,L,F=split(e,C,6,7)
   for mode,G in [('live',L),('frozen',F)]:
    for name,mod in modules(e).items():
     mod.weight.requires_grad_(True)
     z=live(e,C['ids']) if mode=='live' else frozen(e,C)
     loss=z[6]-z[7]
     dense=torch.autograd.grad(loss,mod.weight,allow_unused=True)[0]
     mod.weight.requires_grad_(False)
     computed=G[name][0].T@G[name][1] if name in G else torch.zeros_like(mod.weight)
     expected=torch.zeros_like(mod.weight) if dense is None else dense
     torch.testing.assert_close(computed,expected,rtol=3e-4,atol=3e-6,msg=f'{kind}/{mode}/{name}')
   for row in rows:
    self.assertLess(row['closure_error'],1e-10)
    if row['family'] in ['gate','q','k']:self.assertEqual(row['content_sq'],0.)

 def test_frozen_finite_affinity_and_live_finite_difference(self):
  for kind in ['llama','qwen2','olmo2']:
   e,C=tiny(kind);_,L,F=split(e,C,6,7);base=float(C['z'][6]-C['z'][7])
   for name,mod in modules(e).items():
    u,v=random_one(mod.weight.shape,5)
    with torch.no_grad():
     with patch(mod,u,v,.03):zf=frozen(e,C)
     change=float(zf[6]-zf[7])-base
     self.assertAlmostEqual(change,.03*directional(F.get(name),u,v),delta=3e-6)
     eps=.001
     with patch(mod,u,v,eps):plus=live(e,C['ids']);pl=float(plus[6]-plus[7])
     with patch(mod,u,v,-eps):minus=live(e,C['ids']);mi=float(minus[6]-minus[7])
    self.assertAlmostEqual((pl-mi)/(2*eps),directional(L.get(name),u,v),delta=3e-4)

 def test_origin_certificates(self):
  for kind in ['llama','qwen2','olmo2']:
   e,C=tiny(kind);rows,vectors=origins(e,C,[1,2],1e-4)
   self.assertEqual(vectors.shape,(2,2,16));self.assertLess(max(r['certificate'] for r in rows),1e-4)

 def test_gradient_algebra(self):
  torch.manual_seed(4);a=(torch.randn(4,9),torch.randn(4,7));b=(torch.randn(4,9),torch.randn(4,7))
  A=a[0].double().T@a[1].double();B=b[0].double().T@b[1].double()
  self.assertAlmostEqual(dot(subtract(a,b),subtract(a,b)),float((A-B).square().sum()),places=5)
  u,v=rank_one(a);sv=torch.linalg.svdvals(A)[0]
  self.assertAlmostEqual(directional(a,u,v),float(sv),places=4)
  U=torch.linalg.qr(torch.randn(9,3,dtype=torch.float64))[0];V=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64))[0]
  self.assertAlmostEqual(projected_energy(a,U,V),float((U.T@A@V).square().sum()),places=5)
  D=torch.randn(9,7);self.assertAlmostEqual(delta_dot(a,D),float((A*D).sum()),places=5)

 def test_true_fisher_score_identity(self):
  e,C=tiny('llama');name='0:gate';mod=modules(e)[name];p=C['z'].double().softmax(-1);mean=torch.zeros_like(mod.weight,dtype=torch.float64)
  total=0.
  for y in range(len(p)):
   obj=lambda z,y=y:z.log_softmax(-1)[y]
   L,_=factors(e,C,obj,'live');F,_=factors(e,C,obj,'frozen')
   d,x=L[name];G=d.double().T@x.double();mean+=p[y]*G
   total+=float(p[y])*stats(L[name],F.get(name))['live_sq']
  self.assertLess(float(mean.norm()),1e-6);self.assertGreater(total,0)
  rows=fisher(e,C,3,7);self.assertEqual(len(rows),3*14)

 def test_import_analysis_zip(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);r=dict(prompt='Who is Ada?',subject='Ada',answer='a person',aliases=['a person'],id=1,group_id=2,relation='test',split='evaluation',template=0)
   with zipfile.ZipFile(p/'input.zip','w') as z:z.writestr('records.jsonl',json.dumps(dict(path='llama1b/features/fact/t0.json',data=r))+'\n')
   rows=import_prompts(p/'input.zip',p/'prompts.jsonl');self.assertEqual(len(rows),1)
   self.assertNotIn('input_ids',rows[0])

if __name__=='__main__':unittest.main()
