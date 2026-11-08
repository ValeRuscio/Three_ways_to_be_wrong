import types
from collections import Counter
import torch
from revision import build_real as br


def test_answer_matching_boundaries():
    assert br.answer_matches('Annapolis is a city.', ['Annapolis'])
    assert br.answer_matches('The Hague.', ['The Hague'])
    assert not br.answer_matches('I think it is Annapolis', ['Annapolis'])
    assert not br.answer_matches('Yorkshire', ['York'])


def test_subject_partition():
    assert br.partition('same',4)==br.partition('same',4)
    counts=Counter(br.partition(str(i),4) for i in range(5000))
    assert set(counts)=={'inventory','calibration','evaluation'}
    assert 700<counts['inventory']<1300


def test_stratified_sampling():
    rows=[{'id':i,'s_pop':i} for i in range(19)]
    sampled=br.stratified_order(rows,42)
    assert sorted(r['id'] for r in sampled)==list(range(19))
    assert sampled==br.stratified_order(rows,42)


def test_aliases():
    assert br.aliases_for({'obj':'A','possible_answers':'["B", "A"]'})==['A','B']


class Tokenizer:
    eos_token_id=0
    def __call__(self,text,return_tensors=None,return_offsets_mapping=False,add_special_tokens=True):
        import re
        words=list(re.finditer(r'\S+',text))
        ids=[{'Paris':10,'London':11}.get(m.group(),2) for m in words]
        result=types.SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)
        result.offset_mapping=[(m.start(),m.end()) for m in words]
        return result
    def decode(self,ids,skip_special_tokens=True):
        return 'Paris' if int(ids[0])==10 else 'London'


class Model:
    def __init__(self,top): self.top=top
    def generate(self,ids,**kw):
        scores=torch.zeros((1,20)); scores[0,self.top]=2
        return types.SimpleNamespace(sequences=torch.cat([ids,torch.tensor([[self.top]])],1),scores=[scores])


def test_generation_metadata():
    row=dict(id=1,subj_id=4,subj='France',prop='capital',obj='Paris',possible_answers='["Paris"]',s_pop=10)
    good=br.evaluate(Model(10),Tokenizer(),row,'cpu',24)
    bad=br.evaluate(Model(11),Tokenizer(),row,'cpu',24)
    assert good['status']=='eligible' and good['record']['correct']
    assert bad['status']=='eligible' and not bad['record']['correct']
    assert bad['record']['target_first_token']==10
    assert bad['record']['competitor_token']==11
    assert bad['record']['excluded_decoy_tokens']==[10]


def test_alternative_alias_initial_excluded():
    row=dict(id=1,subj_id=4,subj='France',prop='capital',obj='Paris',possible_answers='["Paris", "London"]')
    r=br.evaluate(Model(11),Tokenizer(),row,'cpu',24)
    assert r['status']=='correct_alternative_initial_token'


def test_real_builder_end_to_end_without_external_downloads(monkeypatch,tmp_path):
    import sys, json, re
    dataset=[]
    for ri,rel in enumerate(br.TEMPLATES):
        for i in range(200):
            k=ri*200+i
            dataset.append(dict(id=k,subj_id=k,subj=f'Person{k}',prop=rel,obj=f'Answer{k}',
                                possible_answers=json.dumps([f'Answer{k}']),s_pop=k))
    class T:
        eos_token_id=0
        def __call__(self,text,return_tensors=None,return_offsets_mapping=False,add_special_tokens=True):
            words=list(re.finditer(r'\S+',text)); ids=[]
            for m in words:
                word=m.group().strip('.,:')
                if word.startswith('Person'): value=10000+int(word[6:])
                elif word.startswith('Answer'): value=20000+int(word[6:])
                else: value=2
                ids.append(value)
            return types.SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids,
                                         offset_mapping=[(m.start(),m.end()) for m in words])
        def decode(self,ids,skip_special_tokens=True):
            n=int(ids[0]); return f'Answer{n-20000}' if n>=20000 else 'Wrong'
    class M:
        def to(self,d): return self
        def eval(self): return self
        def parameters(self): return []
        def __call__(self,ids,**kw):
            k=[int(n)-10000 for n in ids[0] if 10000<=int(n)<20000][-1]
            target=20000+k
            scores=torch.zeros((1,len(ids[0]),22000)); scores[0,-1,target]=10
            return types.SimpleNamespace(logits=scores)
        def generate(self,ids,**kw):
            k=[int(n)-10000 for n in ids[0] if 10000<=int(n)<20000][-1]
            top=20000+k if k%2==0 else 3
            scores=torch.zeros((1,22000)); scores[0,top]=10
            return types.SimpleNamespace(sequences=torch.cat([ids,torch.tensor([[top]])],1),scores=[scores])
    monkeypatch.setitem(sys.modules,'datasets',types.SimpleNamespace(load_dataset=lambda *a,**kw:dataset))
    monkeypatch.setattr(br.AutoTokenizer,'from_pretrained',lambda *a,**kw:T())
    monkeypatch.setattr(br.AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:M())
    monkeypatch.setattr(sys,'argv',['build_real','--model','fake','--dataset-revision','fake',
        '--out',str(tmp_path),'--device','cpu','--calibration-successes-per-relation','2',
        '--evaluation-per-relation','12','--ranking-examples','2'])
    br.main()
    records=[json.loads(l) for l in (tmp_path/'cohort.jsonl').read_text().splitlines()]
    cal={r['group_id'] for r in records if r['split']=='calibration'}
    evaluation={r['group_id'] for r in records if r['split']=='evaluation'}
    assert len(cal)==12 and not cal&evaluation
    assert all(br.partition(r['group_id'],0)==r['split'] for r in records)
    assert (tmp_path/'decoys.json').exists()
    assert len(json.loads((tmp_path/'donor_coverage.json').read_text()))==2
