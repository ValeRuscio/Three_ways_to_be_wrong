DEFAULT = dict(
 seed=0, device='cuda', tolerance=1e-4, max_new_tokens=32,
 models=[{'key':'llama1b','repo':'meta-llama/Llama-3.2-1B'},
         {'key':'llama3b','repo':'meta-llama/Llama-3.2-3B'},
         {'key':'llama8b','repo':'meta-llama/Llama-3.1-8B'},
         {'key':'qwen7b','repo':'Qwen/Qwen2.5-7B'}],
 stages=['capture','analysis','weights','curves','sink','training'],
 ranks=[1,4,16,64], doses=[.001,.003,.01,.03], seeds=[0,1,2],
 scopes=['below','above','all'], crossover_fraction=11/16,
 fit_calibration_pairs=128, ridge=1e-4, specificity_n=8,
 curve_k=[0,1,4,16,32,128,512,2048,8192,32768,131072],
 curve_orders=['contribution','random'], curve_modes=['live','frozen'],
 sink_doses=[1.,.75,.5,.25,0.], sink_heads=4,
 source_quantile=.05, log_ratio_epsilon=1e-8,
 training_models=[{'key':'olmo1b','repo':'allenai/OLMo-2-0425-1B'},
                  {'key':'olmo7b','repo':'allenai/OLMo-2-1124-7B'}],
 training_fractions=[.01,.05,.1,.2,.4,.6,.8,1.], training_sustained=2,
 training_facts_jsonl=None, gradient_manifest=None,
 bootstrap=2000, permutations=100000, max_eval=None,
)

def validate(c):
 if set(c)-set(DEFAULT):raise ValueError('Unknown configuration keys: '+str(set(c)-set(DEFAULT)))
 if not 0<c['crossover_fraction']<1:raise ValueError('crossover_fraction must be between 0 and 1')
 if any(x<=0 for x in c['ranks']+c['doses']):raise ValueError('Ranks and doses must be positive')
 if not set(c['stages'])<=set(DEFAULT['stages']):raise ValueError('Unknown stage')
 if not set(c['scopes'])<={'below','above','all'}:raise ValueError('Unknown scope')
 if not set(c['curve_modes'])<={'live','frozen'}:raise ValueError('Unknown curve mode')
 if not set(c['curve_orders'])<={'contribution','random'}:raise ValueError('Unknown curve order')
 if any(not 0<=x<=1 for x in c['sink_doses']):raise ValueError('Invalid sink dose')
 if c['max_eval'] is not None and c['max_eval']<1:raise ValueError('Invalid max_eval')
