import json,copy
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast,AutoTokenizer,AutoModelForCausalLM
from causal_suite.config import preset,spec
from causal_suite.engine import live
from causal_suite.data import render
from test_engine import make


def test_real_runner_resume_and_protocol_guard(tmp_path,monkeypatch):
 import causal_suite.run as runner
 vocab={'[UNK]':0,'[EOS]':1,**{'word'+str(i):i for i in range(2,96)}};backend=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));backend.pre_tokenizer=Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='[EOS]');e=make()
 rows=[]
 for i in range(35):
  split='inventory' if i<25 else 'calibration' if i<30 else 'evaluation'
  r=dict(id=str(i),group_id=str(i),subject='word'+str(3+i),relation='capital',split=split)
  top=int(live(e,torch.tensor(tok(render(r)).input_ids)).argmax());answer='word'+str(40+i) if split=='inventory' else tok.convert_ids_to_tokens(top)
  r.update(answer=answer,aliases=[answer]);rows.append(r)
 rows.append(dict(id='missing_subject',group_id='missing_subject',subject=None,relation='capital',split='evaluation',answer='word40',aliases=['word40']))
 facts=tmp_path/'facts.jsonl';facts.write_text('\n'.join(json.dumps(r) for r in rows));cfg=preset();cfg.update(facts_jsonl=str(facts),device='cpu',run_repairs=False,run_planted=False,bootstrap=8,permutations=8,max_new_tokens=2,min_calibration=2)
 s=spec('test','unused');s['sha']='test';cfg['models']=[{k:v for k,v in s.items() if k!='sha'}]
 monkeypatch.setattr(runner,'resolve_models',lambda cfg,out:[s]);monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**kw:tok);monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:e.model)
 out=tmp_path/'result';runner.run(cfg,out)
 assert len(list((out/'test/raw/records').glob('*.json')))==10
 invalid=json.loads((out/'test/raw/invalid_data_rows.json').read_text())
 assert invalid['count']==1 and invalid['excluded'][0]['id']=='missing_subject'
 def unexpected(*args,**kwargs):raise AssertionError('Resume should not repeat measurement')
 monkeypatch.setattr(runner,'measure',unexpected);runner.run(cfg,out)
 import pytest
 altered=copy.deepcopy(cfg);altered['seed']+=1
 with pytest.raises(RuntimeError,match='changed'):runner.run(altered,out)
