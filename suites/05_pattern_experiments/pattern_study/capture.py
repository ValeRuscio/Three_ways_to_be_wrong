from pathlib import Path
import numpy as np
import torch
from causal_suite.data import TEMPLATES,span,target_tokens
from gate_study.factors import extract,mlp_input
from .common import *


def prepare(tok,r,t,cal):
 templates=TEMPLATES[r['relation']]
 text=(templates[t] if t<3 else 'Question: '+templates[1]).format(s=r['subject'])
 if t==4:
  candidates=sorted([q for q in cal if q['relation']==r['relation'] and str(q['group_id'])!=str(r['group_id'])],key=lambda q:stamp(q['id']))
  used=set();examples=[]
  for q in candidates:
   if q['group_id'] in used:continue
   used.add(q['group_id']);examples.append(templates[1].format(s=q['subject'])+' '+q['answer'])
   if len(examples)==3:break
  if len(examples)<3:raise ValueError('Fewer than three independent few-shot examples')
  text='\n\n'.join(examples+[templates[1].format(s=r['subject'])])
 token_ids=tok(text).input_ids;S=span(tok,text,r['subject']);g=target_tokens(tok,text,r['answer'])[0]
 return dict(id=str(r['id']),group_id=str(r['group_id']),relation=r['relation'],split=r['split'],template=t,
  subject=r['subject'],answer=r['answer'],aliases=r['aliases'],prompt=text,input_ids=token_ids,source=S,g=g)


def screen(e,tok,r,t,cal,cfg):
 try:m=prepare(tok,r,t,cal)
 except (ValueError,TypeError,KeyError) as ex:return dict(id=str(r['id']),group_id=str(r['group_id']),split=r['split'],relation=r['relation'],template=t,status='excluded',reason=str(ex))
 z=logits(e,m);other=z.clone();other[m['g']]=-torch.inf;m['c']=int(other.argmax());m.update(endpoint(z,m));m.update(generation(e,tok,m,cfg))
 # Strict pair selection, but unconditional full-answer endpoint also retained.
 m.update(correct=bool(m['gold_top1'] and m['full_answer_correct']),status='measured')
 return m


def capture_one(e,tok,r,t,cal,root,cfg):
 p=cache_path(root,r,t);mp=p.with_suffix('.json')
 if mp.exists():
  m=load(mp)
  if m['status']!='measured' or p.exists():return m
 m=screen(e,tok,r,t,cal,cfg)
 if m['status']!='measured':dump(mp,m);return m
 C=e.capture(ids(e,m),tol=cfg['tolerance']);a=extract(e,C,m['source'],m['g'],m['c'],cfg['tolerance'])
 # Cache only arrays used by these analyses; no full GPU programs are serialized.
 keep={k:a[k] for k in ['pre','gate','up_subject','down_gold','attention_rows','attention_roles','head_margin']}
 keep['down_margin']=a['down_gold']-a['down_competitor']
 with torch.no_grad():keep['x']=torch.stack([mlp_input(e,C,l,d['x'])[[m['source'][-1],len(m['input_ids'])-1]] for l,d in enumerate(C['layers'])]).cpu().numpy()
 ledger=e.ledger(C,m['source'],[m['g'],m['c']],[],cfg['tolerance'])
 m.update(certificate=C['certificate'],factor_certificate=float(a['factor_certificate'].max()),
  origins={k:ledger[str(m['g'])]['origins'][k]-ledger[str(m['c'])]['origins'][k] for k in ledger[str(m['g'])]['origins']})
 npz(p,**keep);dump(mp,m);return m


def run(e,tok,records,root,cfg,templates=range(5)):
 cal=[r for r in records if r['split']=='calibration']
 for i,r in enumerate(records):
  for t in templates:capture_one(e,tok,r,t,cal,root,cfg)
  print(f'Capture {i+1}/{len(records)} facts saved',flush=True)
 dump(Path(root)/'tokenizer.json',dict(vocabulary_hash=stamp(sorted(tok.get_vocab().items())),backend_hash=stamp(tok.backend_tokenizer.to_str()),repo=tok.name_or_path))
