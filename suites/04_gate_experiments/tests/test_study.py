import json
from pathlib import Path
import numpy as np
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,Olmo2Config,Olmo2ForCausalLM
from causal_suite.engine import Engine
from gate_study.common import DEFAULT,dump,npz,cache_path,meta_path,key
from gate_study.factors import extract,mlp_input,role_mass
from gate_study.variance import crossed
from gate_study.edits import weight_patch,make_arms,state,endpoints
from gate_study.routing import patch_row,pattern_patch,live_with_attentions
from gate_study.activation import preactivation_patch
PAIRS=[(LlamaConfig,LlamaForCausalLM),(Qwen2Config,Qwen2ForCausalLM),(Olmo2Config,Olmo2ForCausalLM)]


def make(config=LlamaConfig,cls=LlamaForCausalLM):
 torch.manual_seed(47);cfg=config(vocab_size=96,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2);cfg._attn_implementation='eager';model=cls(cfg).float().eval()
 return Engine(model)

@pytest.mark.parametrize('config,cls',PAIRS)
def test_factorization_by_finite_neuron_deletion(config,cls):
 e=make(config,cls);C=e.capture(torch.tensor([5,4,7,8,2]));S=[1,2];g,c=3,4;a=extract(e,C,S,g,c)
 assert a['factor_certificate'].max()<1e-4
 assert np.allclose(a['attention_roles'].sum(-1),1,atol=1e-6)
 l=0;site=0;pos=S[-1];n=int(np.argmax(np.abs(a['contribution_gold'][l,site])))
 modified=dict(C,layers=[dict(d) for d in C['layers']]);modified['layers'][l]['gate']=C['layers'][l]['gate'].clone();modified['layers'][l]['gate'][pos,n]=0
 def contribution(P):
  s=e.component(P,S)[-1];zero=torch.zeros_like(s);return float(e.read(s,P,[g])[0]-e.read(zero,P,[g])[0])
 actual=contribution(C)-contribution(modified)
 assert abs(a['contribution_gold'][l,site,n])>1e-8
 assert actual==pytest.approx(float(a['contribution_gold'][l,site,n]),abs=2e-5)

@pytest.mark.parametrize('config,cls',PAIRS)
def test_live_weight_and_activation_rollback(config,cls):
 e=make(config,cls);C=e.capture(torch.tensor([5,4,7,8,2]));meta=dict(input_ids=C['ids'].tolist(),g=3,c=4);base,_=state(e,meta);x=mlp_input(e,C,1,C['layers'][1]['x'])[-1];pre=e.blocks[1].mlp.gate_proj(x).detach()
 neurons=[1,2];direction=x/x.norm();delta=.1*torch.ones(2,1)/np.sqrt(2)*direction[None]
 before=e.blocks[1].mlp.gate_proj.weight.clone()
 with weight_patch(e,1,'gate',neurons,delta):
  got=e.blocks[1].mlp.gate_proj(x).detach();assert torch.allclose(got[neurons]-pre[neurons],delta@x,atol=1e-6)
  changed,_=state(e,meta);assert not torch.allclose(changed,base,atol=1e-8)
 assert torch.equal(e.blocks[1].mlp.gate_proj.weight,before)
 with pytest.raises(RuntimeError):
  with weight_patch(e,1,'down',neurons,torch.randn(32,2)*.01):raise RuntimeError('Intentional failure')
 restored,_=state(e,meta);assert torch.allclose(restored,base,atol=1e-6)
 with preactivation_patch(e,1,4,neurons,torch.ones(2)*.2):
  changed,_=state(e,meta);assert not torch.allclose(changed,base,atol=1e-8)
 restored,_=state(e,meta);assert torch.allclose(restored,base,atol=1e-6)

@pytest.mark.parametrize('config,cls',PAIRS)
def test_pattern_simplex_identity_and_rollback(config,cls):
 e=make(config,cls);C=e.capture(torch.tensor([5,4,7,8,2]));meta=dict(input_ids=C['ids'].tolist(),g=3,c=4)
 for operation in ['template','subject','sink','role_mass']:
  with pattern_patch(e,C,[1,2],[1,2],[(0,0)],operation):assert torch.allclose(live_with_attentions(e,meta),C['z'],atol=1e-5)
 row=torch.tensor([.2,.3,.1,.2,.2]);donor=torch.tensor([.4,.1,.2,.1,.2]);new=patch_row(row,donor,[1,2],[1,2],'template')
 assert float(new.sum())==pytest.approx(1.)
 assert torch.allclose(new[[3,4]],donor[[3,4]])
 assert float(new[1]/new[2])==pytest.approx(float(row[1]/row[2]))


def test_crossed_variance_known_components():
 subjects=np.array([-2.,0.,2.]);templates=np.array([-1.,0.,1.]);X=(subjects[:,None]+templates[None,:])[...,None];a=crossed(X)
 assert a['subject'][0]==pytest.approx(24);assert a['template'][0]==pytest.approx(6);assert a['interaction'][0]<1e-20
 X=(subjects[:,None]*templates[None,:])[...,None];a=crossed(X);assert a['subject'][0]<1e-20 and a['template'][0]<1e-20 and a['interaction'][0]>0


def fixture(root):
 e=make();C=e.capture(torch.tensor([5,4,7,8,2]));g=int(C['z'].argmin());c=int(C['z'].argmax());records=[]
 for i in range(4):
  cal=i<3;gg=c if cal else g;cc=g if cal else c
  r=dict(id=str(i),group_id=str(i),relation='capital',subject='subject',answer='answer',aliases=['answer'],split='calibration' if cal else 'evaluation',eligible=True,labels={'full':'correct' if cal else 'source'})
  records.append(r)
  for t in range(4):
   a=extract(e,C,[1,2],gg,cc)
   if cal:a['pre']=a['pre']+1. # synthetic calibration reference ensures an editable deficit
   npz(cache_path(root,r,t),**a)
   dump(meta_path(root,r,t),dict(id=r['id'],group_id=r['group_id'],relation=r['relation'],split=r['split'],label='correct' if cal else 'source',template=t,status='measured',correct=cal,
    input_ids=C['ids'].tolist(),source=[1,2],g=gg,c=cc,aliases=['answer'],margin=float(C['z'][gg]-C['z'][cc]),target_rank=1 if cal else 96,certificate=C['certificate'],factor_certificate=float(a['factor_certificate'].max()),origin_margin={'subject':0.,'template':0.,'first_token':0.,'bias':0.}))
 return e,records


def test_full_core_pipeline_and_resume(tmp_path,monkeypatch):
 import gate_study.run as runner
 e,records=fixture(tmp_path)
 class Tok:pass
 monkeypatch.setattr(runner.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tok())
 monkeypatch.setattr(runner.AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:e.model)
 cfg=dict(DEFAULT,device='cpu',save_frozen_cache=False,top_k=2,profile_k=[2],sites=['answer'],minimum_reference_subjects=1,weight_relative_norms=[.01],activation_alphas=[.5],control_seeds=[0],specificity_per_case=1,handoff_heads=1,routing_donors=1,generate_on_gold_top1=False,bootstrap=10,permutations=10,run_training=False)
 src=dict(spec=dict(repo='fixture',key='fixture',sha='fixture'),records=records)
 runner.core(src,tmp_path,cfg)
 paths=list((tmp_path/'weight_edits').glob('*/*gate_target*.json'));assert paths
 for p in paths:
  d=json.loads(p.read_text());assert d['delta_norm']>0 and d['preactivation_gain']>0
 stamps={str(p):p.stat().st_mtime_ns for p in paths}
 runner.core(src,tmp_path,cfg)
 assert all(Path(p).stat().st_mtime_ns==stamp for p,stamp in stamps.items())
 assert (tmp_path/'reports/weight_paired_tests_holm.csv').exists()
 assert (tmp_path/'reports/activation_patches.csv').exists()


def test_training_intervals_and_fixed_neurons(tmp_path):
 from gate_study.training import summarize,first_sustained
 assert first_sustained([False,True,False,True,True],2)==3
 specs=[dict(step=i*10+10) for i in range(4)];dump(tmp_path/'checkpoints.json',specs)
 for j,s in enumerate(specs):
  dump(tmp_path/f'step_{s["step"]}/x.json',dict(id='x',group_id='x',relation='capital',step=s['step'],sha=str(j),meta=dict(status='measured',correct=j>=1,top1=j>=1,margin=j-1),
   rows=[dict(set_kind='independent_final_calibration',site='subject',layer=0,neuron=1,pre=j,up_cosine=j*.1,down_cosine=j*.2)]))
 summarize(tmp_path,dict(DEFAULT,bootstrap=10))
 import pandas as pd
 event=pd.read_csv(tmp_path/'reports/training_events.csv').iloc[0]
 assert event.newly_correct and event.first_sustained_step==20 and event.previous_observed_step==10


def test_gradient_geometry_separates_basis_and_probe_steps(tmp_path):
 from gate_study.gradients import build,analyze
 G=np.array([[3.,0.,0.],[0.,2.,0.],[0.,0.,1.],[1.,0.,0.]])
 npz(tmp_path/'a.npz',gate_layer_0=G);npz(tmp_path/'b.npz',gate_layer_0=G)
 m=dict(gradient_origin='pretraining',provenance='Synthetic unit-test gradients, not scientific data',repo='test',layers=[0],basis_snapshots=[dict(step=1,file='a.npz')],probe_snapshots=[dict(step=2,file='b.npz')])
 dump(tmp_path/'input.json',m);build(tmp_path/'input.json',tmp_path/'bases',1)
 dump(tmp_path/'training/fixed_neurons.json',{'x':dict(kind='independent_final_calibration',sites=[dict(layer=0,neurons=[0])])})
 analyze(tmp_path/'bases/manifest.json',tmp_path/'training',dict(DEFAULT,training_repo='test',control_seeds=[0]))
 import pandas as pd
 d=pd.read_csv(tmp_path/'training/reports/training_gradient_alignment.csv');assert d[d.group=='selected'].subspace_energy_fraction.iloc[0]==pytest.approx(1.)
 m['probe_snapshots'][0]['step']=1;dump(tmp_path/'bad.json',m)
 with pytest.raises(ValueError):build(tmp_path/'bad.json',tmp_path/'bad',1)


def test_fresh_capture_tokenization_and_saved_gates(tmp_path,monkeypatch):
 from tokenizers import Tokenizer
 from tokenizers.models import WordLevel
 from tokenizers.pre_tokenizers import Whitespace
 from transformers import PreTrainedTokenizerFast
 from gate_study.cache import capture_case
 words=['[UNK]','[EOS]','The','capital','of','is','subject']+[f'word{i}' for i in range(7,96)]
 backend=Tokenizer(WordLevel({w:i for i,w in enumerate(words)},unk_token='[UNK]'));backend.pre_tokenizer=Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='[EOS]')
 e=make();ids=torch.tensor(tok('The capital of subject is').input_ids);C=e.capture(ids);g=int(C['z'].argmax());assert g not in [0,1]
 answer=tok.decode([g]);r=dict(id='new',group_id='new',relation='capital',subject='subject',answer=answer,aliases=[answer],split='evaluation',labels={'full':'correct'})
 cfg=dict(DEFAULT,device='cpu',max_new_tokens=2)
 m=capture_case(e,tok,r,0,tmp_path,cfg);assert m['status']=='measured' and m['correct']
 saved=torch.load(tmp_path/'frozen_cache'/key(r)/'t0.pt',weights_only=True)
 assert 'gate' in saved['layers'][0] and saved['layers'][0]['gate'].dtype==torch.float32
 monkeypatch.setattr(e,'capture',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Repeated completed capture')))
 assert capture_case(e,tok,r,0,tmp_path,cfg)==m


def test_training_runner_resume_without_downloads(tmp_path,monkeypatch):
 import gate_study.training as training
 from causal_suite.data import digest
 e,records=fixture(tmp_path/'final_reference');ev=records[-1]
 fixed={ev['id']:dict(kind='independent_final_calibration',sites=[dict(layer=0,site='subject',site_index=0,neurons=[1,2])])}
 dump(tmp_path/'fixed_neurons.json',fixed);dump(tmp_path/'tokenizer.json',dict(hash=digest([('x',1)])))
 specs=[dict(repo='test',revision=f'stage1-step{i}-tokens1B',step=i,sha=str(i)) for i in [10,20,30]]
 def pin(cfg,path):dump(path,specs);return specs
 class Tok:
  def get_vocab(self):return {'x':1}
 monkeypatch.setattr(training,'pin',pin);monkeypatch.setattr(training,'load',lambda *a:(Tok(),e))
 monkeypatch.setattr(training,'prepare',lambda tok,r,t:json.loads(meta_path(tmp_path/'final_reference',r,t).read_text()))
 cfg=dict(DEFAULT,device='cpu',bootstrap=10)
 training.run(records,tmp_path,cfg)
 assert len(list(tmp_path.glob('step_*/*.json')))==3
 monkeypatch.setattr(training,'load',lambda *a:(_ for _ in ()).throw(AssertionError('Reloaded completed training checkpoint')))
 training.run(records,tmp_path,cfg)
 assert (tmp_path/'reports/training_neurons.csv').exists()
