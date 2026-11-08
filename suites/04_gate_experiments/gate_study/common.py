import hashlib,json,zipfile,os
from pathlib import Path
import numpy as np
import pandas as pd
from causal_suite.data import digest,save
from causal_suite.stats import boot,holm,paired_p

DEFAULT=dict(seed=0,device='cuda',bootstrap=2000,permutations=2000,tolerance=1e-4,
 save_frozen_cache=True,run_variance=True,run_profiles=True,run_activation_patches=True,activation_alphas=[.25,.5,1.],run_weight_edits=True,run_routing=True,run_training=True,
 max_examples=None,top_k=32,profile_k=[8,32,128],sites=['subject','answer'],minimum_reference_subjects=3,
 weight_relative_norms=[.003,.01,.03],control_seeds=[0,1,2],specificity_per_case=8,
 max_new_tokens=32,generate_on_gold_top1=True,preactivation_levels=[-2.,0.,2.],
 handoff_heads=4,routing_donors=2,
 training_repo='allenai/OLMo-2-0425-1B',training_steps=[20000,100000,190000,380000,760000,1140000,1530000,1907359],
 training_revisions=None,training_sustained=2,training_progress_levels=[.25,.5,.75],
 gradient_manifest=None)


def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def dump(path,obj):save(path,clean(obj))
def table(rows,path):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);Path(path).parent.mkdir(parents=True,exist_ok=True);d.to_csv(path,index=False);return d

def npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with open(tmp,'wb') as f:np.savez_compressed(f,**arrays)
 tmp.replace(path)

def key(r):return digest(str(r['id']))[:24]
def estimate(x,groups,cfg):return boot(x,groups,cfg['bootstrap'],cfg['seed'])
def source_hash():
 root=Path(__file__).parent.parent
 return digest({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for name in ['gate_study','causal_suite'] for p in sorted((root/name).glob('*.py'))})

def source(path,out):
 path=Path(path).expanduser().resolve()
 if path.is_file():
  sha=hashlib.sha256(path.read_bytes()).hexdigest();dest=Path(out)/('input_'+sha[:12]);dest.mkdir(parents=True,exist_ok=True)
  if not (dest/'DONE').exists():
   with zipfile.ZipFile(path) as z:
    for info in z.infolist():
     p=(dest/info.filename).resolve()
     if dest.resolve() not in p.parents:raise ValueError('Unsafe archive path')
    z.extractall(dest)
   (dest/'DONE').write_text(sha)
  path=dest
 result=[]
 for p in sorted(path.glob('*/*/measurements.json')):
  spec=json.loads((p.parent/'model.json').read_text())
  if Path(spec['key']).name!=spec['key'] or spec['key'] in ['.','..']:raise ValueError('Unsafe model key')
  if p.parent.name!='raw':continue
  records=json.loads(p.read_text());result.append(dict(spec=spec,records=records,hash=hashlib.sha256(p.read_bytes()).hexdigest()))
 if not result:raise ValueError('Need original causal_v1 results with model/raw/measurements.json. The compact followup ZIP alone is insufficient.')
 return result


def validate(cfg):
 unknown=set(cfg)-set(DEFAULT)
 if unknown:raise ValueError(f'Unknown settings: {unknown}')
 if cfg['top_k']<1 or cfg['minimum_reference_subjects']<1:raise ValueError('Positive selection sizes required')
 if any(s not in ['subject','answer'] for s in cfg['sites']):raise ValueError('Unknown site')
 if any(v<=0 for v in cfg['weight_relative_norms']):raise ValueError('Weight doses must be positive')
 if cfg['max_examples'] is not None and cfg['max_examples']<1:raise ValueError('Invalid example limit')
 if cfg['training_sustained']<1:raise ValueError('Invalid sustained-success length')


def load_arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def cache_path(root,r,template):return Path(root)/'features'/key(r)/f't{template}.npz'
def meta_path(root,r,template):return cache_path(root,r,template).with_suffix('.json')
def read_meta(root,r,template):return json.loads(meta_path(root,r,template).read_text())

def evaluation(records,cfg):
 out=[r for r in records if r.get('eligible') and r['split']=='evaluation']
 return out if cfg['max_examples'] is None else out[:cfg['max_examples']]
