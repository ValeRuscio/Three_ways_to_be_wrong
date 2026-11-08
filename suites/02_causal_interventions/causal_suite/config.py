"""Presets are explicit; full suite is many GPU-days, not the default."""
def spec(key,repo,family=None,stage=None,revision='main',step=None,origins=True,repairs=False):
 return dict(key=key,repo=repo,family=family or key,stage=stage or key,revision=revision,step=step,origins=origins,repairs=repairs)

def preset(name='one_model'):
 cfg=dict(run_name='causal_v1',seed=0,device='cuda',evaluation_per_relation=100,
 calibration_candidates_per_relation=250,calibration_reference_n=60,min_calibration=20,
 bootstrap=2000,permutations=2000,max_new_tokens=32,consecutive=2,quantile=.1,
 handoff_quantile=.1,alphas=[.25,.5,1.],gammas=[1.25,1.5,2.,4.],selection_doses=[.25,.5,1.,2.,4.],
 layer_fractions=[.25,.5,.75],top_heads=8,chat_check=False,run_repairs=True,run_planted=True,
 repair_limit=None,plant_limit=100,save_frozen_cache=False,facts_jsonl=None,exposure_csv=None,
 # Optional cross-family dose comparison: interpolate within observed common
 # ranges of final pre-norm residual displacement, never compare alpha/gamma directly.
 common_cost_grid=[.01,.03,.1,.3,1.])
 pairs=[('llama32','meta-llama/Llama-3.2-3B','meta-llama/Llama-3.2-3B-Instruct'),
 ('llama31','meta-llama/Llama-3.1-8B','meta-llama/Llama-3.1-8B-Instruct'),
 ('qwen3','Qwen/Qwen2.5-3B','Qwen/Qwen2.5-3B-Instruct'),('qwen7','Qwen/Qwen2.5-7B','Qwen/Qwen2.5-7B-Instruct')]
 alignment=[]
 for fam,a,b in pairs:
  alignment += [spec(fam+'_base',a,fam,'base'),spec(fam+'_instruct',b,fam,'instruct')]
 alignment += [spec('olmo_'+s,'allenai/OLMo-2-1124-7B'+suffix,'olmo',''+s) for s,suffix in [('base',''),('sft','-SFT'),('dpo','-DPO'),('instruct','-Instruct')]]
 training=[spec('pythia_'+str(s),'EleutherAI/pythia-2.8b','pythia','checkpoint','step'+str(s),s,False) for s in [1000,4000,16000,32000,64000,96000,143000]]
 if name=='one_model':models=[spec('llama1b','meta-llama/Llama-3.2-1B',repairs=True)]
 elif name=='alignment':models=alignment
 elif name=='training':models=training;cfg['discover_olmo_training']=True
 elif name=='full':
  models=alignment+training;cfg['discover_olmo_training']=True
  # Six explicitly named core models; all evaluation failures AND successes.
  for m in models:m['repairs']=m['key'] in ['llama32_base','llama32_instruct','llama31_base','qwen3_base','qwen7_base','olmo_base']
 else:raise ValueError('Choose one_model, alignment, training, or full')
 cfg['models']=models;return cfg
