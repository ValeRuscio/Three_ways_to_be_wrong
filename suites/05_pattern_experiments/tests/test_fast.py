import numpy as np
import pytest
import torch
from test_patterns import ARCH,model,meta
from pattern_study.common import ids,logits,load
from pattern_study.fast_curves import FastCurves,skip_prefix_layers
from pattern_study.interventions import frozen

@pytest.mark.parametrize('config,cls',ARCH)
def test_cached_curves_equivalence_and_no_cache_mutation(config,cls,tmp_path):
 e=model(config,cls);m=meta();C=e.capture(ids(e,m));f=FastCurves(e,C,m);before=[(k.clone(),v.clone()) for k,v in f.legacy]
 assert f.check(tmp_path/'validation.json')
 assert all(r['max_logit_error']<1e-5 for r in load(tmp_path/'validation.json')['checks'])
 for l in range(e.L):
  gs={l:(np.arange(32),np.zeros(32,dtype=np.float32))};row=C['layers'][l]['A'][:,-1].cpu().numpy().copy();row[:,0]*=.3;row/=row.sum(-1,keepdims=True);rs={l:row}
  for gs,rs in [(gs,{}),({},rs),(gs,rs)]:
   assert torch.allclose(f.cached_live(gs,rs),f.full_live(gs,rs),atol=2e-6,rtol=1e-5)
   assert torch.allclose(f.cached_frozen(gs,rs),frozen(e,C,gs,rs),atol=2e-6,rtol=1e-5)
 assert all(torch.equal(k,k0) and torch.equal(v,v0) for (k,v),(k0,v0) in zip(f.legacy,before))
 assert torch.allclose(logits(e,m),C['z'],atol=1e-6)
 with pytest.raises(RuntimeError):
  with skip_prefix_layers(e,C,1):raise RuntimeError('deliberate')
 assert 'forward' not in e.blocks[0].__dict__
 assert torch.allclose(logits(e,m),C['z'],atol=1e-6)

@pytest.mark.parametrize('config,cls',ARCH)
def test_final_only_single_token(config,cls,tmp_path):
 e=model(config,cls);m=meta();m['input_ids']=[3];C=e.capture(ids(e,m));f=FastCurves(e,C,m)
 assert f.check(tmp_path/'v.json')


def test_upgrade_preserves_completed_files_and_rejects_config_change(tmp_path):
 from pathlib import Path
 from pattern_study.upgrade import migrate,ORIGINAL_HASH
 from pattern_study.common import dump
 out=tmp_path/'out';source=tmp_path/'new'
 for folder in ['pattern_study','gate_study','causal_suite']:
  (source/folder).mkdir(parents=True);(source/folder/'__init__.py').write_text('new')
 (out/'source/pattern_study').mkdir(parents=True);(out/'source/pattern_study/old.py').write_text('old')
 file=out/'llama1b/curves/abc/DONE.json';dump(file,{'status':'complete'});before=file.read_bytes()
 old=dict(code_hash=ORIGINAL_HASH,config={'seed':0},input_hash='facts');new=dict(old,code_hash='newhash')
 dump(out/'protocol.json',old)
 with pytest.raises(ValueError):migrate(out,dict(new,config={'seed':1}),source)
 assert load(out/'protocol.json')==old
 assert migrate(out,new,source);assert file.read_bytes()==before
 assert (out/'source_versions'/ORIGINAL_HASH/'pattern_study/old.py').read_text()=='old'
 assert not migrate(out,new,source)
