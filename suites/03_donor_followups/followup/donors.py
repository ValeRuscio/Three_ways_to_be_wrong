import random
import torch
from causal_suite.data import digest,span,target_tokens,matches,TEMPLATES
from .frozen import align_positions


def same_layout(r,d):
 if len(r['input_ids'])!=len(d['input_ids']) or r['source']!=d['source']:return False
 S=set(r['source'])
 return all(a==b for i,(a,b) in enumerate(zip(r['input_ids'],d['input_ids'])) if i not in S)


def layout_donors(r,records,cfg):
 candidates=[d for d in records if d.get('eligible') and d['split']=='calibration' and d['relation']==r['relation'] and d['group_id']!=r['group_id'] and same_layout(r,d)]
 choices={}
 for name,correct in [('success',True),('failure_control',False)]:
  selected=[d for d in candidates if d['correct']==correct];selected.sort(key=lambda d:digest([cfg['seed'],r['id'],d['id']]))
  choices[name]=selected[:cfg['donors_per_case']]
 return choices


@torch.no_grad()
def paraphrase_donor(engine,tok,r,cfg):
 if r.get('mode','raw')!='raw':return None,None,[{'reason':'Raw-mode paraphrase templates only; no silent chat substitution'}]
 templates=TEMPLATES[r['relation']][1:]+['Question: '+TEMPLATES[r['relation']][1]]
 audit=[]
 for template in templates[:cfg['paraphrase_templates']]:
  prompt=template.format(s=r['subject'])
  try:
   ids=tok(prompt,return_tensors='pt').input_ids[0].to(engine.device);S=span(tok,prompt,r['subject']);g=target_tokens(tok,prompt,r['answer'])[0]
   if g!=r['g']:raise ValueError('Different target token boundary')
   d=dict(r,prompt=prompt,input_ids=ids.tolist(),source=S)
   mapping=align_positions(r,d)
  except ValueError as e:audit.append(dict(prompt=prompt,status=str(e)));continue
  z=engine.model(ids[None],attention_mask=torch.ones_like(ids[None]),use_cache=False).logits[0,-1]
  if int(z.argmax())!=g:audit.append(dict(prompt=prompt,status='wrong_first_token'));continue
  generation=engine.model.generate(ids[None],attention_mask=torch.ones_like(ids[None]),max_new_tokens=32,do_sample=False,temperature=1.,top_p=1.,pad_token_id=tok.eos_token_id)
  text=tok.decode(generation[0,len(ids):],skip_special_tokens=True)
  if not matches(text,r['aliases']):audit.append(dict(prompt=prompt,status='wrong_full_answer',generated=text));continue
  audit.append(dict(prompt=prompt,status='verified',generated=text,aligned_positions=len(mapping),receiver_positions=len(r['input_ids']),mapping=mapping))
  return engine.capture(ids),mapping,audit
 return None,None,audit
