import numpy as np,pytest
from causal_suite.stats import holm,boot,auc_test,trend_test
from causal_suite.data import random_span,build_fixed,save,matches
import random,json

def test_cluster_bootstrap_not_pseudoreplicated():
 a=boot([0,0,1,1],['a','a','b','b'],100,0);b=boot([0,1],['a','b'],100,0)
 assert a['groups']==2;assert a['lo']==b['lo'] and a['hi']==b['hi']
def test_holm_missing_and_order():assert np.allclose(holm([.03,.001,.04,float('nan')])[:3],[.06,.003,.06])
def test_auc_missing_class():assert auc_test([0,0],[1,2],['a','b'],10)['auc'] is None
def test_span_no_overlap():
 r=random_span(10,[3,4],random.Random(0));assert len(r)==2 and not set(r)&{3,4}
def test_word_boundary():assert matches('Paris, France',['Paris']);assert not matches('Parisian',['Paris'])
def test_split_leakage_fails(tmp_path):
 rows=[dict(id=str(i),group_id='same',subject='x',relation='capital',answer='a',aliases=['a'],split=s) for i,s in enumerate(['inventory','calibration','evaluation'])]
 p=tmp_path/'facts.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows))
 with pytest.raises(ValueError,match='leakage'):build_fixed({'facts_jsonl':str(p)},tmp_path/'out')

def test_fixed_effects_equal_model_mean():
 from causal_suite.stats import fixed_effects
 result=fixed_effects([1.,2.,3.,5.,6.,7.],['a']*3+['b']*3,['x','y','z']*2)
 assert abs(result['estimate']-4.)<1e-8
 assert result['lo']<result['estimate']<result['hi']

def test_exposure_nested_names_count(tmp_path):
 from causal_suite.exposure import count
 rows=[dict(id='x',split='evaluation',subject='New York Times',aliases=['New York'])]
 path=tmp_path/'corpus.jsonl';path.write_text(json.dumps({'text':'New York Times'})+'\n')
 values,meta=count(rows,[path]);assert values[0]['exposure']==1 and meta['documents']==1


def test_missing_subject_guard():
 from causal_suite.data import span
 for bad in [None,'', '   ',12]:
  with pytest.raises(ValueError,match='Missing or empty'):span(None,'prompt',bad)
