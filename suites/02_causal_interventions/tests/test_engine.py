import pytest,torch
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,GPTNeoXConfig,GPTNeoXForCausalLM,Olmo2Config,Olmo2ForCausalLM
from causal_suite.engine import Engine,live,residual_patch,attention_scale

PAIRS=[(LlamaConfig,LlamaForCausalLM),(Qwen2Config,Qwen2ForCausalLM),(GPTNeoXConfig,GPTNeoXForCausalLM),(Olmo2Config,Olmo2ForCausalLM)]

def make(config=LlamaConfig,cls=LlamaForCausalLM):
 torch.manual_seed(15);c=config(vocab_size=96,hidden_size=32,intermediate_size=48,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,rotary_pct=.5);c._attn_implementation='eager'
 model=cls(c).float().eval()
 # Nonzero biases exercise affine offsets, including final LayerNorm bias.
 with torch.no_grad():
  for name,p in model.named_parameters():
   if name.endswith('bias'):p.uniform_(-.03,.03)
 return Engine(model)

@pytest.mark.parametrize('config,cls',PAIRS)
def test_native_ledger_pullback_and_handoff(config,cls):
 e=make(config,cls);ids=torch.tensor([1,4,7,8,2]);C=e.capture(ids);S=[1,2];tokens=[4,7,11];led=e.ledger(C,S,tokens,[3,4])
 for t in tokens:
  assert abs(sum(led[str(t)]['origins'].values())-float(live(e,ids)[t]))<1e-4
  assert max(led[str(t)]['total_by_cut'])-min(led[str(t)]['total_by_cut'])<1e-4
 # Linearity checks against the actual affine layer, not a second copy of equations.
 x=torch.randn_like(C['layers'][0]['x']);y=torch.randn_like(x)
 assert torch.allclose(e.linear(x+y,0,C),e.linear(x,0,C)+e.linear(y,0,C),atol=1e-4,rtol=1e-4)
 assert e.head_scores(C,S,4).shape==(3,4)

@pytest.mark.parametrize('config,cls',PAIRS)
def test_live_edits_identity_change_and_reversal(config,cls):
 e=make(config,cls);ids=torch.tensor([1,4,7,8,2]);base=live(e,ids)
 with attention_scale(e,[1,2],[(1,0)],1.):assert torch.allclose(live(e,ids),base,atol=1e-5)
 with attention_scale(e,[1,2],[(1,0)],2.):
  edited=live(e,ids);assert not torch.allclose(edited,base,atol=1e-8);C=e.capture(ids);e.ledger(C,[1,2],[4],[])
 assert torch.allclose(live(e,ids),base)
 with residual_patch(e,1,[1,2],torch.randn(2,32)*.1):
  edited=live(e,ids);assert not torch.allclose(edited,base,atol=1e-8);C=e.capture(ids);e.ledger(C,[1,2],[4],[])
 assert torch.allclose(live(e,ids),base)


def test_subject_first_position_partition():
 e=make();C=e.capture(torch.tensor([1,4,7,8,2]));r=e.ledger(C,[0,1],[4],[3,4])['4']
 assert r['origins']['first_token']==0
 assert abs(sum(r['origins'].values())-float(C['z'][4]))<1e-5


def test_full_repair_and_plant_grid():
 from causal_suite.config import preset
 from causal_suite.repairs import execute_grid,planted
 e=make();C=e.capture(torch.tensor([1,4,7,8,2]));g=int(C['z'].argmax());c=int(C['z'].topk(2).indices[1]);pool=[t for t in range(96) if t!=g][:25]
 r=dict(id='a',group_id='a',relation='capital',g=g,c=c,source=[1,2],correct=True,labels={'full':'correct'},pool_tokens=pool,tau_threshold=0,share_threshold=0)
 cfg=preset();cfg.update(alphas=[.5,1.],gammas=[1.5,2.],selection_doses=[.5,1.],layer_fractions=[.5],top_heads=2)
 rows=execute_grid(e,r,C,cfg,(C,[1,2]),(C,[3,4]));assert set(x['family'] for x in rows)=={'source','transport','selection','selection_upstream','transport_secondary'}
 assert all(x['final_normalized_relative_cost']>=0 for x in rows)
 result,audit=planted(e,r,C,cfg);assert len(audit)==3
 selection=next(x for x in audit if x['operation']=='selection');assert selection['failed'];assert selection['exact_reversal_recovered']
 assert torch.allclose(live(e,C['ids']),C['z'])


def test_real_measurement_path_without_downloads():
 from tokenizers import Tokenizer
 from tokenizers.models import WordLevel
 from tokenizers.pre_tokenizers import Whitespace
 from transformers import PreTrainedTokenizerFast
 from causal_suite.measure import measure,assign_labels
 from causal_suite.config import preset
 vocab={'[UNK]':0,'[EOS]':1,**{'word'+str(i):i for i in range(2,96)}}
 backend=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));backend.pre_tokenizer=Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='[EOS]')
 e=make();prompt='word4 word7 word8';ids=torch.tensor(tok(prompt).input_ids);top=int(live(e,ids).argmax())
 assert top not in [0,1,4,7,8]
 r=dict(id='x',group_id='x',subject='word7',relation='capital',answer='word'+str(top),aliases=['word'+str(top)],split='evaluation')
 cfg=preset();cfg['max_new_tokens']=2;result,C=measure(e,tok,r,prompt,['word'+str(i) for i in range(2,96)],None,cfg)
 assert result['eligible'] and result['correct'];assert len(result['hard_tokens'])==20;assert 'ledger' in result
 assert result['random_span'] is not None
