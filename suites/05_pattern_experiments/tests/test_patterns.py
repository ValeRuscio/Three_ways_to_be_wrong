import json
import numpy as np
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,Olmo2Config,Olmo2ForCausalLM
from causal_suite.engine import Engine
from gate_study.factors import extract,mlp_input
from pattern_study.common import *
from pattern_study.config import DEFAULT
from pattern_study import lowrank,interventions,analysis,weights,curves,sink,report,training

ARCH=[(LlamaConfig,LlamaForCausalLM),(Qwen2Config,Qwen2ForCausalLM),(Olmo2Config,Olmo2ForCausalLM)]

def model(config=LlamaConfig,cls=LlamaForCausalLM):
 torch.manual_seed(12);c=config(vocab_size=64,hidden_size=24,intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2);c._attn_implementation='eager'
 return Engine(cls(c).float().eval())

def meta():return dict(input_ids=[3,5,8,9,2],g=4,c=7)

@pytest.mark.parametrize('config,cls',ARCH)
@pytest.mark.parametrize('kind',['gate','up','down'])
def test_lowrank_equals_physical_edit_and_rollback(config,cls,kind):
 e=model(config,cls);m=meta();base=logits(e,m);w=getattr(e.blocks[0].mlp,kind+'_proj').weight;u=torch.randn(w.shape[0],2)*.01;v=torch.randn(w.shape[1],2)*.01;before=w.clone()
 with lowrank.apply(e,kind,{0:(u,v)}):got=logits(e,m)
 with torch.no_grad():w.add_(u@v.T)
 physical=logits(e,m)
 with torch.no_grad():w.copy_(before)
 assert torch.allclose(got,physical,atol=2e-6)
 assert torch.equal(logits(e,m),base)
 with pytest.raises(RuntimeError):
  with lowrank.apply(e,kind,{0:(u,v)}):raise RuntimeError('deliberate')
 assert torch.equal(logits(e,m),base)
 assert lowrank.frob(u,v)==pytest.approx(float((u@v.T).norm()),rel=1e-5)

@pytest.mark.parametrize('config,cls',ARCH)
@pytest.mark.parametrize('kind',['gate','up','down'])
def test_live_gradient_matches_parameter_autograd(config,cls,kind):
 e=model(config,cls);m=meta();f=lowrank.gradients(e,m,kind,[0,1]);w=getattr(e.blocks[0].mlp,kind+'_proj').weight
 w.requires_grad_(True);x=ids(e,m);z=e.model(x[None],use_cache=False).logits[0,-1];g=torch.autograd.grad(z[m['g']]-z[m['c']],w)[0];w.requires_grad_(False)
 assert torch.allclose(f[0][0]@f[0][1].T,g.cpu(),atol=2e-6,rtol=1e-4)
 u,v=lowrank.compress(*f[0],64);assert torch.allclose(u@v.T,g.cpu(),atol=2e-6,rtol=1e-4)


def test_rank_identifiability_and_fit():
 torch.manual_seed(2);X=torch.randn(9,1);D=torch.randn(13,1);u,v=lowrank.fit(X,D,64,1e-8)
 assert u.shape[1]==1;assert torch.allclose((u@v.T)@X,D,atol=1e-5)
 X=torch.randn(9,6);D=torch.randn(13,6);u,v=lowrank.fit(X,D,4,1e-5);assert u.shape[1]==4
 assert torch.linalg.matrix_rank(u@v.T,atol=1e-4)==4

@pytest.mark.parametrize('config,cls',ARCH)
def test_gate_attention_identity_and_frozen_identity(config,cls):
 e=model(config,cls);m=meta();C=e.capture(ids(e,m));changes={l:(np.arange(32),d['gate'][-1].cpu().numpy()) for l,d in enumerate(C['layers'])}
 with interventions.gates(e,changes,4):assert torch.allclose(logits(e,m),C['z'],atol=2e-6)
 assert torch.allclose(interventions.frozen(e,C,changes),C['z'],atol=2e-6)
 rows={l:d['A'][:,-1].cpu().numpy() for l,d in enumerate(C['layers'])}
 with interventions.attention(e,rows):assert torch.allclose(logits(e,m,True),C['z'],atol=2e-6)
 zero={0:(np.arange(32),np.zeros(32,dtype=np.float32))}
 with interventions.gates(e,zero,4):assert not torch.allclose(logits(e,m),C['z'],atol=1e-7)
 assert torch.allclose(logits(e,m),C['z'],atol=2e-6)


def test_mass_transport_and_log_ratios():
 rng=np.random.default_rng(1);a=rng.random((2,4,7));a/=a.sum(-1,keepdims=True);b=rng.random((2,4,5));b/=b.sum(-1,keepdims=True)
 out=interventions.align_rows(a,b,[2,3],[2]);assert np.allclose(out.sum(-1),1)
 assert np.allclose(out[...,2],a[...,2:4].sum(-1))
 assert np.allclose(interventions.align_rows(a,a,[2,3],[2,3]),a)
 f=rng.normal(size=(30,3));s=rng.normal(size=(30,3));f[0,0]=0
 d=analysis.factor_summary(f,s,1e-8);assert d['excluded_neurons']==1;assert d['log_identity_max_error']<1e-12
 assert sum(d[k+'_absolute_change_share'] for k in ['gate','up_subject','readout'])==pytest.approx(1.)


def make_cache(root):
 e=model();records=[]
 for i in range(5):
  r=dict(id=str(i),group_id=str(i),subject='X',answer='Y',aliases=['Y'],relation='capital',split='calibration' if i<3 else 'evaluation');records.append(r)
  for t in range(4):
   m=dict(r,input_ids=[3,5+i,8,9,2+t],source=[1],g=4,c=7,template=t,status='measured',correct=t==1 or i==4,margin=-1.,origins=dict(subject=.2,template=-1.2,first_token=0.,bias=0.))
   C=e.capture(ids(e,m));a=extract(e,C,[1],4,7);keep={k:a[k] for k in ['pre','gate','up_subject','down_gold','attention_rows','attention_roles','head_margin']}
   keep['down_margin']=a['down_gold']-a['down_competitor'];keep['x']=torch.stack([mlp_input(e,C,l,d['x'])[[1,4]].detach() for l,d in enumerate(C['layers'])]).numpy()
   m['certificate']=C['certificate'];m['factor_certificate']=float(a['factor_certificate'].max())
   npz(cache_path(root,r,t),**keep);dump(meta_path(root,r,t),m)
 measured.cache_clear();return e,records


def test_full_cached_pipeline_and_resume(tmp_path,monkeypatch):
 e,records=make_cache(tmp_path);cfg=dict(DEFAULT,device='cpu',ranks=[1,4],doses=[.01],seeds=[0],scopes=['all'],fit_calibration_pairs=2,specificity_n=1,curve_k=[0,1,4],bootstrap=10,permutations=20,sink_doses=[1.,0.],curve_orders=['contribution'],curve_modes=['live','frozen'])
 monkeypatch.setattr(weights,'generation',lambda *a:dict(generated='Y',full_answer_correct=True))
 analysis.run(tmp_path,cfg);weights.run(e,None,tmp_path,cfg);curves.run(e,tmp_path,cfg);sink.run(e,tmp_path,cfg);report.run(tmp_path,cfg)
 files=list(tmp_path.glob('weights/*/*.json'));assert files
 measured_files=[p for p in files if load(p).get('status')=='measured']
 for p in measured_files:
  d=load(p);assert d['actual_delta_norm']==pytest.approx(d['budget'],rel=1e-5)
 before={p:p.stat().st_mtime_ns for p in files};weights.run(e,None,tmp_path,cfg)
 assert all(p.stat().st_mtime_ns==t for p,t in before.items())
 assert (tmp_path/'reports/half_effect_sizes.csv').exists()


def test_sign_test_cluster_and_event_censor():
 assert report.sign_test([1,1,1],[0,1,2],100,0)==pytest.approx(.25)
 assert report.sign_test([1,1,1],[0,0,0],100,0)==1.
 assert training.first_sustained([False,True,False,True,True],2)==3
 assert training.first_sustained([False,False,True],2) is None


def test_real_capture_tokenization_and_resume(tmp_path):
 from tokenizers import Tokenizer,models,pre_tokenizers
 from transformers import PreTrainedTokenizerFast
 from pattern_study.capture import capture_one
 tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'[EOS]':1,'The':2,'capital':3,'of':4,'X':5,'is':6,'Y':7},unk_token='[UNK]'));tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',eos_token='[EOS]');e=model();r=dict(id='a',group_id='a',subject='X',answer='Y',aliases=['Y'],relation='capital',split='evaluation')
 cfg=dict(DEFAULT,device='cpu',max_new_tokens=1)
 m=capture_one(e,tok,r,0,[],tmp_path,cfg);assert m['status']=='measured';assert m['certificate']<1e-4
 a=load_arrays(cache_path(tmp_path,r,0));assert a['x'].shape==(2,2,24)
 old=cache_path(tmp_path,r,0).stat().st_mtime_ns;capture_one(e,tok,r,0,[],tmp_path,cfg);assert cache_path(tmp_path,r,0).stat().st_mtime_ns==old


def test_training_own_selection_and_resume(tmp_path,monkeypatch):
 import shutil
 source=tmp_path/'fixture';e,records=make_cache(source);root=tmp_path/'training'
 specs=[dict(repo='toy',step=i,sha=str(i)) for i in [1,2]]
 def pins(spec,cfg,path):dump(path,specs);return specs
 monkeypatch.setattr(training,'checkpoints',pins);monkeypatch.setattr(training,'broad_cohort',lambda *a:records)
 class Backend:
  def to_str(self):return 'tokenizer'
 class Tok:backend_tokenizer=Backend()
 monkeypatch.setattr(training,'model_load',lambda *a:(Tok(),e))
 monkeypatch.setattr(training,'screen',lambda e,tok,r,*a:dict(correct=True,id=r['id']))
 def capture(e,tok,records,folder,cfg,templates):
  shutil.copytree(source/'features',Path(folder)/'features')
  for r in records:
   p=meta_path(folder,r,0);m=load(p);m['correct']=str(folder).endswith('step_2') or r['id']=='4';dump(p,m)
 monkeypatch.setattr(training,'capture',capture)
 cfg=dict(DEFAULT,device='cpu');training.run(dict(key='toy',repo='toy'),root,cfg)
 events=__import__('pandas').read_csv(root/'reports/training_events.csv');assert len(events)==2
 assert load(root/'selected_facts.json')['selected']=={'calibration':3,'evaluation':2}
 assert load(root/'reports/gradient_status.json')['status']=='not_run'
 stamp0=(root/'step_1/CAPTURE_COMPLETE.json').stat().st_mtime_ns
 training.run(dict(key='toy',repo='toy'),root,cfg)
 assert stamp0==(root/'step_1/CAPTURE_COMPLETE.json').stat().st_mtime_ns
