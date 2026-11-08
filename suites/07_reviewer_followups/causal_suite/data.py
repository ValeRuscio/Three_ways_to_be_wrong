"""Fixed, subject-disjoint prompts; no outcome-balanced evaluation sampling."""
import hashlib,json,re,random
from pathlib import Path

TEMPLATES={
 'capital':['The capital of {s} is','What is the capital of {s}? Answer:','{s} has its capital at'],
 'capital of':['{s} is the capital of','Which country has {s} as its capital? Answer:','The country whose capital is {s} is'],
 'author':['The author of "{s}" is','Who wrote "{s}"? Answer:','"{s}" was written by'],
 'composer':['The composer of "{s}" is','Who composed "{s}"? Answer:','"{s}" was composed by'],
 'director':['The director of "{s}" is','Who directed "{s}"? Answer:','"{s}" was directed by'],
 'place of birth':['{s} was born in','Where was {s} born? Answer:','The birthplace of {s} is']}


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def normalize(x):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',x.casefold())).strip()
def matches(text,answers):
 text=normalize(text)
 return any(text==normalize(a) or text.startswith(normalize(a)+' ') for a in answers if normalize(a))
def span(tok,prompt,subject,add_special_tokens=True):
 if not isinstance(subject,str) or not subject.strip():raise ValueError('Missing or empty subject')
 start=prompt.rfind(subject)
 if start<0:raise ValueError('Subject missing')
 offsets=tok(prompt,return_offsets_mapping=True,add_special_tokens=add_special_tokens).offset_mapping
 s=[i for i,(a,b) in enumerate(offsets) if a<start+len(subject) and b>start]
 if not s:raise ValueError('Empty subject span')
 return s

def target_tokens(tok,prompt,answer,chat=False):
 if chat:
  ids=tok(answer.strip(),add_special_tokens=False).input_ids
 else:
  p=tok(prompt).input_ids; full=tok(prompt+' '+answer.strip()).input_ids
  if full[:len(p)]!=p:raise ValueError('Answer changes prompt tokenization')
  ids=full[len(p):]
 if not ids:raise ValueError('Empty answer')
 return ids

def render(row,template=0,chat_tokenizer=None):
 text=TEMPLATES[row['relation']][template].format(s=row['subject'])
 if chat_tokenizer is not None:
  text=chat_tokenizer.apply_chat_template([{'role':'user','content':text+'\nGive only the answer.'}],tokenize=False,add_generation_prompt=True)
 return text

def build_fixed(config,out):
 path=Path(out)/'fixed_data.json'
 if path.exists():return json.loads(path.read_text())
 if config.get('facts_jsonl'):
  rows=[json.loads(x) for x in Path(config['facts_jsonl']).read_text().splitlines() if x.strip()]
  required={'id','group_id','subject','relation','answer','aliases','split'}
  if not rows or any(not required.issubset(r) for r in rows):raise ValueError('Facts JSONL schema: '+str(required))
  result={'rows':rows,'source':'user supplied; preserve fixed split'}
 else:
  from datasets import load_dataset
  from huggingface_hub import HfApi
  revision=HfApi().dataset_info('akariasai/PopQA').sha
  raw=load_dataset('akariasai/PopQA',revision=revision,split='test');unique={}
  for r in sorted(raw,key=lambda r:str(r['id'])):
   if r['prop'] not in TEMPLATES:continue
   key=(str(r['subj_id']),r['prop']);unique.setdefault(key,r)
  groups={}
  for r in unique.values():
   k=int(digest([str(r['subj_id']),config['seed']])[:12],16)%100
   split='inventory' if k<20 else 'calibration' if k<60 else 'evaluation'
   aliases=r['possible_answers'];aliases=json.loads(aliases) if isinstance(aliases,str) else aliases
   record=dict(id=str(r['id']),group_id=str(r['subj_id']),subject=r['subj'],relation=r['prop'],answer=r['obj'],
      aliases=list(dict.fromkeys([r['obj']]+aliases)),split=split,popularity=int(r.get('s_pop') or 0))
   groups.setdefault((split,r['prop']),[]).append(record)
  rows=[]
  for (split,rel),rs in sorted(groups.items()):
   rs.sort(key=lambda r:digest([config['seed'],r['id']]))
   n=len(rs) if split=='inventory' else config['calibration_candidates_per_relation'] if split=='calibration' else config['evaluation_per_relation']
   rows.extend(rs[:n])
  result=dict(rows=rows,source='akariasai/PopQA',revision=revision)
 ids=[r['id'] for r in result['rows']]
 if len(ids)!=len(set(ids)):raise ValueError('IDs must be unique')
 seen={}
 for r in result['rows']:
  if r['relation'] not in TEMPLATES:raise ValueError('Provide templates in data.py for new relation')
  if r['split'] not in ('inventory','calibration','evaluation'):raise ValueError('Invalid split')
  if r['group_id'] in seen and seen[r['group_id']]!=r['split']:raise ValueError('Subject leakage across splits')
  seen[r['group_id']]=r['split']
 if not all(any(r['split']==s for r in result['rows']) for s in ['inventory','calibration','evaluation']):raise ValueError('All three splits required')
 save(path,result);return result

def inventory(rows):
 return {rel:sorted({a for r in rows if r['relation']==rel and r['split']=='inventory' for a in r['aliases']}) for rel in TEMPLATES}

def random_span(T,S,rng):
 choices=[list(range(i,i+len(S))) for i in range(T-len(S)+1) if not set(range(i,i+len(S)))&set(S)]
 return rng.choice(choices) if choices else None
