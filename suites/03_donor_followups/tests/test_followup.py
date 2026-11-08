import json
import numpy as np
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,GPTNeoXConfig,GPTNeoXForCausalLM,Olmo2Config,Olmo2ForCausalLM
from causal_suite.engine import Engine
from followup.common import DEFAULT
from followup.frozen import hybrid,gradients,apply_edit,realize,edge_scores,align_positions,aligned_coefficients,greedy
from followup.sink import vector_snapshot,aggregate_snapshots,frequency_vector
from followup.geometry import analyze,build_bases,load_manifest
from followup.report import report
PAIRS=[(LlamaConfig,LlamaForCausalLM),(Qwen2Config,Qwen2ForCausalLM),(GPTNeoXConfig,GPTNeoXForCausalLM),(Olmo2Config,Olmo2ForCausalLM)]


def engine(config=LlamaConfig,cls=LlamaForCausalLM):
 torch.manual_seed(15);c=config(vocab_size=96,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,rotary_pct=.5);c._attn_implementation='eager';m=cls(c).float().eval()
 with torch.no_grad():
  for name,p in m.named_parameters():
   if name.endswith('bias'):p.uniform_(-.03,.03)
 return Engine(m)

@pytest.mark.parametrize('config,cls',PAIRS)
def test_hybrid_certificates_and_exact_finite_edits(config,cls):
 e=engine(config,cls);C=e.capture(torch.tensor([1,4,7,8,2]));D=e.capture(torch.tensor([1,6,9,8,2]))
 assert torch.allclose(hybrid(e,C,D)['z'],C['z'],atol=1e-5)
 for policy in ['recipient','donor']:
  for A,G in [(1,0),(0,1),(1,1)]:
   H=hybrid(e,C,D,A,G,policy);ledger=e.ledger(H,[1,2],[4,7],[])
   for t in [4,7]:assert sum(ledger[str(t)]['origins'].values())==pytest.approx(float(H['z'][t]),abs=1e-5)
 C,grad=gradients(e,C,4,7,'joint');base=float(C['z'][4]-C['z'][7])
 for l in range(e.L):
  ix=(4,3);old=float(C['layers'][l]['gate'][ix]);new=old+.7
  changed=realize(e,apply_edit(C,dict(layer=l,kind='gate',index=ix,value=new)))
  assert float(changed['z'][4]-changed['z'][7])-base==pytest.approx(.7*float(grad[l]['gate'][ix]),abs=2e-5)
  A=C['layers'][l]['A'];desired=A.clone();ix=(1,4,2);desired[ix]=.6
  predicted=float(edge_scores(A,desired,grad[l]['A'])[ix]);changed=realize(e,apply_edit(C,dict(layer=l,kind='edge',index=ix,value=.6)))
  assert float(changed['z'][4]-changed['z'][7])-base==pytest.approx(predicted,abs=2e-5)
  assert torch.allclose(changed['layers'][l]['A'].sum(-1),torch.ones_like(A.sum(-1)),atol=1e-6)

@pytest.mark.parametrize('config,cls',PAIRS)
def test_full_vocab_origins_and_geometry(config,cls,tmp_path):
 e=engine(config,cls);C=e.capture(torch.tensor([5,4,7,8,2]));D=e.capture(torch.tensor([5,6,9,8,2]));cfg=dict(DEFAULT)
 for i,X in enumerate([C,D]):
  vector_snapshot(e,X,[1,2],tmp_path/'vectors'/f'{i}.npz',dict(id=str(i),group_id=str(i),relation='capital',template_key='t',label='source'),cfg)
 rows,summaries=aggregate_snapshots(tmp_path/'vectors',tmp_path/'reports')
 assert len(rows)==6 and len(summaries)==3
 with np.load(tmp_path/'vectors/0.npz') as a,np.load(tmp_path/'vectors/1.npz') as b:
  assert np.allclose(sum(a.values()),C['z'].numpy(),atol=1e-5)
  xx=np.stack([a['first_token'],b['first_token']]).astype(float);xx-=xx.mean(-1,keepdims=True)
  den=(xx**2).sum();expected=((xx-xx.mean(0))**2).sum()/den if den>1e-20 else None
  observed=next(s for s in summaries if s['origin']=='first_token')['variation_fraction']
  assert observed is None if expected is None else observed==pytest.approx(expected,abs=1e-9)
 Q=np.linalg.qr(np.random.default_rng(5).normal(size=(32,5)))[0]
 data=analyze(e,C,[1,2],4,7,[(0,1,Q)],cfg)
 for r in data:
  assert r['delivered_margin']==pytest.approx(r['projected_margin']+r['orthogonal_margin'],abs=2e-5)
  assert 0<=r['local_energy_fraction']<=1.00001


def test_alignment_search_and_report(tmp_path):
 e=engine();C=e.capture(torch.tensor([1,4,7,8,2]));D=e.capture(torch.tensor([1,3,4,7,8,2]));r=dict(input_ids=C['ids'].tolist(),source=[1,2]);d=dict(input_ids=D['ids'].tolist(),source=[2,3]);mapping=align_positions(r,d)
 assert mapping[1]==2 and mapping[2]==3 and mapping[4]==5
 target=aligned_coefficients(C,D,mapping)
 g=int(C['z'].argmin());c=int(C['z'].argmax())
 result=greedy(e,C,target,g,c,budget=3,prune=True)
 assert len(result['trace'])<=4 and result['stop'] in ['positive_margin','budget_exhausted','no_improving_single_edit']
 p=tmp_path/'edit_search/x';p.mkdir(parents=True)
 meta=dict(id='a',group_id='a',label='source',regime='small_subject_margin',mode='joint',control='greedy')
 (p/'joint_greedy.json').write_text(json.dumps(dict(result,**meta)))
 report(tmp_path,dict(DEFAULT,bootstrap=10,permutations=10,edit_budget=3))
 assert (tmp_path/'reports/edit_distance_joint.pdf').exists()


def test_gradient_builder_and_manifest_rejection(tmp_path):
 G=np.diag([3.,2.,1.,.5]);np.savez(tmp_path/'grad.npz',layer_0=G)
 m=dict(gradient_origin='pretraining',provenance='synthetic unit test, not scientific data',model_sha='abc',num_heads=2,layers=[0],snapshots=['grad.npz'])
 (tmp_path/'input.json').write_text(json.dumps(m));build_bases(tmp_path/'input.json',tmp_path/'bases',1)
 loaded=load_manifest(tmp_path/'bases/manifest.json','abc',4)
 assert abs(loaded[0][2][0,0])==pytest.approx(1.)
 with pytest.raises(ValueError):load_manifest(tmp_path/'bases/manifest.json','wrong',4)
 np.savez(tmp_path/'freq.npz',counts=np.arange(96));(tmp_path/'freq.json').write_text(json.dumps(dict(tokenizer_hash='good')))
 with pytest.raises(ValueError):frequency_vector(tmp_path/'freq.npz',96,'abc','bad')
 f,_=frequency_vector(tmp_path/'freq.npz',96,'abc','good');assert np.linalg.norm(f)==pytest.approx(1.)


def test_verified_recovery_and_pruning():
 e=engine();C=e.capture(torch.tensor([1,4,7,8,2]));g=int(C['z'].argmin());c=int(C['z'].argmax());C,gr=gradients(e,C,g,c,'gate');margin=float(C['z'][g]-C['z'][c])
 target=aligned_coefficients(C,C,{i:i for i in range(5)})
 for t in target:t['gate_mask'].zero_();t['A_mask'].zero_()
 l=1;ix=tuple(int(i) for i in np.unravel_index(int(gr[l]['gate'].abs().argmax()),gr[l]['gate'].shape));slope=float(gr[l]['gate'][ix]);assert abs(slope)>1e-8
 target[l]['gate'][ix]+=(-margin+.2)/slope;target[l]['gate_mask'][ix]=True
 result=greedy(e,C,target,g,c,budget=2,mode='gate',prune=True)
 assert result['recovered_margin'] and result['edit_count']==1 and result['pruned_count']==1
 assert result['trace'][-1]['margin']==pytest.approx(.2,abs=2e-5)


def test_full_runner_and_resume_without_downloads(tmp_path,monkeypatch):
 import followup.run as runner
 from followup.common import dump
 e=engine();ids=[1,4,7,8,2];C=e.capture(torch.tensor(ids));g=int(C['z'].argmin());c=int(C['z'].argmax())
 def record(i,split,correct):
  return dict(id=str(i),group_id=str(i),relation='capital',subject='subject',prompt='capital of subject',input_ids=ids,source=[1,2],split=split,correct=correct,eligible=True,g=g,c=c,margin=float(C['z'][g]-C['z'][c]),ledger=e.ledger(C,[1,2],[g,c],[]),labels={'full':'correct' if correct else 'source'})
 records=[record(0,'evaluation',False),record(1,'evaluation',True),record(2,'calibration',True),record(3,'calibration',False)]
 class Tok:
  def get_vocab(self):return {str(i):i for i in range(96)}
 monkeypatch.setattr(runner.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tok())
 monkeypatch.setattr(runner.AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:e.model)
 monkeypatch.setattr(runner,'paraphrase_donor',lambda *a,**kw:(C,{i:i for i in range(5)},[{'status':'test_only_verified'}]))
 cfg=dict(DEFAULT,device='cpu',run_cpu=False,edit_budget=2,random_search_seeds=[0],bootstrap=10,permutations=10)
 source=dict(spec=dict(repo='test',key='test',sha='test'),records=records,mode='raw')
 runner.run_gpu(source,tmp_path,cfg)
 assert len(list((tmp_path/'vectors').glob('*.npz')))==2
 hp=next((tmp_path/'hybrids').glob('*.json'));assert len(json.loads(hp.read_text())['rows'])==16
 first=hp.stat().st_mtime_ns
 monkeypatch.setattr(runner.Engine,'capture',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Resume repeated a saved case')))
 runner.run_gpu(source,tmp_path,cfg);assert hp.stat().st_mtime_ns==first
 assert (tmp_path/'reports/hybrid_factorial.pdf').exists()


def test_reference_counts_and_reuse(tmp_path,monkeypatch):
 import sys,types
 import followup.reference_corpus as ref
 class Tok:
  def get_vocab(self):return {'a':0,'b':1,'c':2,'d':3}
  def __call__(self,texts,**kw):return {'input_ids':[[int(v) for v in s.split()] for s in texts]}
 class Data:
  def __len__(self):return 3
  def __getitem__(self,s):return {'text':['0 1 1','2',''][s]}
 class API:
  def dataset_info(self,*args):return types.SimpleNamespace(sha='dataset_sha')
 monkeypatch.setattr(ref,'HfApi',API)
 monkeypatch.setattr(ref,'sources',lambda *args:(tmp_path,[dict(spec=dict(key='m',repo='test',sha='model_sha'))]))
 monkeypatch.setattr(ref,'load_dataset',lambda *args,**kw:Data())
 monkeypatch.setattr(ref.AutoTokenizer,'from_pretrained',lambda *args,**kw:Tok())
 monkeypatch.setattr(ref.AutoConfig,'from_pretrained',lambda *args,**kw:types.SimpleNamespace(vocab_size=4))
 monkeypatch.setattr(sys,'argv',['reference','--input','test','--out',str(tmp_path)])
 ref.main()
 with np.load(tmp_path/'m.npz') as z:assert z['counts'].tolist()==[1,2,1,0]
 monkeypatch.setattr(ref,'load_dataset',lambda *args,**kw:(_ for _ in ()).throw(AssertionError('Recounted existing reference')))
 ref.main()
