import argparse
import numpy as np
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,GPTNeoXConfig,GPTNeoXForCausalLM
from revision.common import run_score,label,validate_records,manifest
from revision.measure import supported,provenance,partition_certificate
from revision.baselines import matched_scores,identity_certificate
from revision.predict import grouped_folds,cv_predict
from revision.robustness import features
from revision.internal import patch,patch_measure


def tiny(kind='llama'):
    torch.manual_seed(4); torch.set_num_threads(1)
    kw=dict(vocab_size=24,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,max_position_embeddings=32,attn_implementation='eager')
    if kind=='llama': model=LlamaForCausalLM(LlamaConfig(**kw,num_key_value_heads=2))
    elif kind=='qwen2': model=Qwen2ForCausalLM(Qwen2Config(**kw,num_key_value_heads=2))
    else: model=GPTNeoXForCausalLM(GPTNeoXConfig(**kw,rotary_pct=1.,hidden_act='gelu'))
    return model.eval()


@pytest.mark.parametrize('kind',['llama','qwen2','gpt_neox'])
def test_partition_and_linearity(kind):
    model=tiny(kind); W,api=supported(model); ids=torch.tensor([2,5,8,4,7])
    C=api.build(ids); assert api.certify(C)<1e-4
    assert partition_certificate(W,C,[1,2])<1e-4
    both=provenance(W,C,[1,2],3)
    separate=provenance(W,C,[1],3)+provenance(W,C,[2],3)
    assert abs(both-separate)<1e-5


def test_readability_same_position():
    assert run_score([[1,-1,1],[-1,1,-1]],2)<0
    assert run_score([[1,2,-1]],2)==1
    assert label(False,100,100,0,0)=='source'
    assert label(True,-1,100,0,0)=='transport'
    assert label(True,0,0,0,0)=='selection'


@pytest.mark.parametrize('kind',['llama','qwen2'])
def test_live_gate_derivative_and_ig(kind):
    model=tiny(kind); W,api=supported(model); ids=torch.tensor([2,5,8,4,7])
    assert identity_certificate(model,W,ids)<1e-4
    scores=matched_scores(model,W,ids,2,3,[1],(0,0),steps=16)
    assert abs(scores['gradient_fd_error'])<2e-4
    assert abs(scores['ig_residual'])<2e-4


def test_donor_patch_derivative_and_cleanup():
    model=tiny(); W,api=supported(model); ids=torch.tensor([2,5,8,4,7])
    d=torch.randn(1,32)*.01
    grad=patch_measure(model,ids,2,3,1,[1],d,.5,True)
    plus=patch_measure(model,ids,2,3,1,[1],d,.501)['margin']
    minus=patch_measure(model,ids,2,3,1,[1],d,.499)['margin']
    assert abs(grad-(plus-minus)/.002)<2e-4
    with pytest.raises(RuntimeError):
        with patch(model,1,[1],d,1.): raise RuntimeError('cleanup')
    assert len(model.model.layers[1]._forward_pre_hooks)==0


def test_group_disjointness():
    groups=np.repeat(np.arange(10),3); seen=[]
    for tr,te in grouped_folds(groups,5,1):
        assert not set(groups[tr])&set(groups[te]); seen.extend(te)
    assert sorted(seen)==list(range(30))


def test_reject_split_leakage():
    a=dict(id='a',group_id='fact',split='calibration',correct=True,source_span=[0,1])
    b=dict(a,id='b',split='evaluation')
    with pytest.raises(ValueError,match='leakage'): validate_records([a,b])


def test_empty_and_near_zero_are_not_success():
    r=dict(id='a',target_logit=1.,variants={'original':dict(lens=[[[1.,2.],[1.,2.]]],
        token_identity=False,legacy_edge_sum=1.,provenance={'0':1.},attention_mass=.1)})
    assert features(r,'original','provenance:0',2,'random',0,2,1e-6) is None
    r['target_logit']=0.
    assert features(r,'original','provenance:0',1,'random',0,2,1e-6) is None


def test_manifest_rejects_changed_config(tmp_path):
    manifest(tmp_path,argparse.Namespace(seed=0))
    with pytest.raises(ValueError,match='another configuration'): manifest(tmp_path,argparse.Namespace(seed=1))


def test_prediction_train_only_preprocessing(monkeypatch):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    observed=[]; original=StandardScaler.fit
    def spy(self,X,y=None,sample_weight=None):
        observed.append(len(X)); return original(self,X,y,sample_weight=sample_weight)
    monkeypatch.setattr(StandardScaler,'fit',spy)
    df=pd.DataFrame({'x':np.arange(20),'model':['m']*20}); y=np.tile([0,1],10)
    pred,fold,notes=cv_predict(df,['x'],['model'],y,np.arange(20),5,0)
    assert observed==[16]*5
    assert np.isfinite(pred).all()


@pytest.mark.parametrize('edited',[False,True])
def test_native_gradxact_runs_with_frozen_parameters(edited):
    from revision.baselines import legacy_gradxact
    model=tiny(); W,api=supported(model); ids=torch.tensor([2,5,8,4,7]); C=api.build(ids)
    from known_circuit2 import decoy_edit
    score=legacy_gradxact(model,W,C,ids,2,3,[[1],[2]],decoy_edit if edited else None)
    assert score.shape==(2,W.L,W.H)
    assert torch.isfinite(score).all()
    assert all(len(b.self_attn.o_proj._forward_pre_hooks)==0 for b in model.model.layers)


def test_holm_known_values():
    from revision.internal_report import holm
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])


@pytest.mark.parametrize('kind',['qwen2','gpt_neox'])
def test_nonzero_bias_partition(kind):
    model=tiny(kind)
    with torch.no_grad():
        if kind=='qwen2':
            for b in model.model.layers: b.self_attn.v_proj.bias.fill_(.03)
        else:
            for b in model.gpt_neox.layers:
                b.attention.dense.bias.fill_(.03)
                b.mlp.dense_4h_to_h.bias.fill_(.02)
    W,api=supported(model); C=api.build(torch.tensor([2,5,8,4,7]))
    assert api.certify(C)<1e-4
    assert partition_certificate(W,C,[1,2])<1e-4
