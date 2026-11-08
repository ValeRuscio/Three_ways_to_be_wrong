import hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from causal_suite.data import digest,save
from causal_suite.stats import boot,holm,paired_p

DEFAULT=dict(seed=0,bootstrap=2000,permutations=2000,shifts=[-40,-20,-10,-5,0,5,10,20,40],
 match_margin_caliper=1.,match_length_caliper=4,match_rank_caliper=10,device='cuda',
 run_cpu=True,run_hybrids=True,run_edit_search=True,run_sink=True,run_subspace=True,
 max_examples=None,donors_per_case=3,hybrid_norms=['recipient','donor'],
 edit_budget=256,edit_modes=['gate','edge','joint'],edit_scope='all',
 pruning=True,random_search_seeds=[0,1,2],paraphrase_templates=3,
 subspace_manifest=None,frequency_file=None,frequency_files={},frequency_source_label=None,sink_vocab_block=8192,
 threshold=.1,geometry_heads=None,certificate_tolerance=1e-4)


def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def dump(p,obj):save(p,clean(obj))
def csv(rows,p):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);Path(p).parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False);return d

def estimate(values,groups,cfg):return boot(values,groups,cfg['bootstrap'],cfg['seed'])
def tests(rows,p):
 d=pd.DataFrame(rows)
 if len(d) and 'p' in d:d['p_holm']=holm(pd.to_numeric(d.p,errors='coerce'))
 return csv(d,p)

def sources(path,work):
 path=Path(path).expanduser().resolve()
 if path.is_file():
  sha=hashlib.sha256(path.read_bytes()).hexdigest();dest=Path(work)/('input_'+sha[:12])
  if not (dest/'EXTRACTION_COMPLETE').exists():
   dest.mkdir(parents=True,exist_ok=True)
   with zipfile.ZipFile(path) as z:
    for info in z.infolist():
     target=(dest/info.filename).resolve()
     if dest.resolve() not in target.parents:raise ValueError('Unsafe archive path')
    z.extractall(dest)
   (dest/'EXTRACTION_COMPLETE').write_text(sha)
  path=dest
 found=[]
 for p in sorted(path.glob('*/*/measurements.json')):
  spec=json.loads((p.parent/'model.json').read_text());records=json.loads(p.read_text())
  found.append(dict(spec=spec,records=records,mode=p.parent.name,path=str(p),source_hash=hashlib.sha256(p.read_bytes()).hexdigest()))
 if not found:raise ValueError('Expected causal result directory or ZIP with model/mode/measurements.json')
 return path,found


def group_label(r):return r.get('labels',{}).get('full','undetermined')
def regime(r,eps=.1):
 if r['correct']:return 'correct'
 s=r['ledger'][str(r['g'])]['origins']['subject']-r['ledger'][str(r['c'])]['origins']['subject']
 return 'subject_favors_wrong' if s < -eps else 'subject_favors_gold_but_loses' if s>eps else 'small_subject_margin'
