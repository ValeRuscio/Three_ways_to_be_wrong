import csv,hashlib,io,json,os,platform,sys,zipfile
from pathlib import Path
import torch
from causal_suite.data import span,target_tokens,matches


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def load(path):return json.loads(Path(path).read_text())
def key(m):return digest([str(m['id']),m['template'],m['prompt']])[:24]


def import_prompts(source,dest):
 source=Path(source);records={};excluded=0
 def accept(d):
  nonlocal excluded
  if not isinstance(d,dict) or not {'prompt','subject','answer','aliases','id','group_id','relation'}.issubset(d):return
  if d.get('split') not in ('evaluation','eval'):return
  if not isinstance(d['subject'],str) or not d['subject'].strip():excluded+=1;return
  if not isinstance(d['aliases'],list) or not d['aliases']:raise ValueError('aliases must be a nonempty list')
  m={k:d[k] for k in ['prompt','subject','answer','aliases','id','group_id','relation']}
  m.update(id=str(m['id']),group_id=str(m['group_id']),template=d.get('template',0),split='evaluation')
  records[key(m)]=m
 if source.is_dir():
  files=source.glob('*/features/*/*.json')
  if (source/'features').exists():files=source.glob('features/*/*.json')
  for p in files:accept(load(p))
 elif source.suffix=='.zip':
  with zipfile.ZipFile(source) as z:
   if 'records.jsonl' in z.namelist():
    with z.open('records.jsonl') as f:
     for line in f:
      item=json.loads(line)
      if '/features/' in '/'+item['path'] and not any(p.startswith(('training_','step_')) for p in Path(item['path']).parts):accept(item['data'])
   else:
    for n in z.namelist():
     if '/features/' in '/'+n and n.endswith('.json') and not any(p.startswith(('training_','step_')) for p in Path(n).parts):accept(json.loads(z.read(n)))
 else:
  for line in source.read_text().splitlines():
   if line.strip():accept(json.loads(line))
 if not records:raise ValueError('No evaluation prompt metadata found. Supply a pattern results folder/ZIP or prompts JSONL with split=evaluation.')
 # Stable hash order, not outcome order. Same fact/template/prompt occurs only once across models.
 rows=[records[k] for k in sorted(records)]
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 dest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 print(f'Imported {len(rows)} prompts / {len(set(r["id"] for r in rows))} facts; invalid subjects excluded={excluded}',flush=True)
 return rows


def pin(spec,path):
 if Path(path).exists():
  saved=load(path)
  if saved['repo']!=spec['repo'] or saved.get('revision','main')!=spec.get('revision','main'):raise ValueError('Pinned model specification changed')
  return saved
 from huggingface_hub import HfApi
 s=dict(spec,sha=HfApi().model_info(spec['repo'],revision=spec.get('revision','main')).sha)
 dump(path,s);return s


def model_load(spec,device):
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from causal_suite.engine import Engine
 tok=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['sha'])
 m=AutoModelForCausalLM.from_pretrained(spec['repo'],revision=spec['sha'],torch_dtype=torch.float32,attn_implementation='eager').to(device)
 return tok,Engine(m)


def tokenize(tok,r,device):
 ids=torch.tensor(tok(r['prompt']).input_ids,device=device)
 g=target_tokens(tok,r['prompt'],r['answer'])[0]
 return ids,span(tok,r['prompt'],r['subject']),g


@torch.no_grad()
def generate(e,tok,ids,aliases,max_new):
 y=e.model.generate(ids[None],attention_mask=torch.ones_like(ids[None]),max_new_tokens=max_new,
  do_sample=False,temperature=1.,top_p=1.,use_cache=True,pad_token_id=tok.eos_token_id)
 text=tok.decode(y[0,len(ids):],skip_special_tokens=True)
 return dict(generated=text,answer_correct=matches(text,aliases))


def label(r,baseline,regimes,model):
 # Factual correctness takes precedence over any historical token-based label.
 if baseline['answer_correct']:return 'answer_success'
 return regimes.get((model,str(r['id']),str(r['template'])),'unlabelled_answer_failure')


def read_regimes(path):
 if not path:return {}
 rows=list(csv.DictReader(open(path)));required={'model','id','template','regime'}
 if any(not required.issubset(r) for r in rows):raise ValueError('Regime CSV requires model,id,template,regime')
 return {(r['model'],str(r['id']),str(r['template'])):r['regime'] for r in rows}


def environment():
 import transformers
 return dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,platform=platform.platform(),
  cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
