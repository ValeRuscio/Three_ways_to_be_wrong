from pathlib import Path
import json,hashlib
import torch
from causal_suite.data import TEMPLATES,span,target_tokens,matches
from .common import cache_path,meta_path,dump,npz,key
from .factors import extract


def prompt_text(r,t):
 templates=TEMPLATES[r['relation']]+['Question: '+TEMPLATES[r['relation']][1]]
 return templates[t].format(s=r['subject'])


def prepare(tok,r,t):
 prompt=prompt_text(r,t);ids=tok(prompt).input_ids;S=span(tok,prompt,r['subject']);g=target_tokens(tok,prompt,r['answer'])[0]
 return dict(id=str(r['id']),group_id=str(r['group_id']),relation=r['relation'],subject=r['subject'],answer=r['answer'],aliases=r['aliases'],split=r['split'],template=t,
  label=r.get('labels',{}).get('full','unlabelled'),prompt=prompt,input_ids=ids,source=S,g=g)


@torch.no_grad()
def generate(engine,tok,ids,aliases,cfg):
 y=engine.model.generate(ids[None],attention_mask=torch.ones_like(ids[None]),max_new_tokens=cfg['max_new_tokens'],do_sample=False,temperature=1.,top_p=1.,pad_token_id=tok.eos_token_id,use_cache=True)
 text=tok.decode(y[0,len(ids):],skip_special_tokens=True)
 return text,matches(text,aliases)


def save_frozen(C,path):
 def cpu(x):
  if isinstance(x,torch.Tensor):return x.detach().cpu()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  if isinstance(x,list):return [cpu(v) for v in x]
  return x
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(cpu(C),tmp);tmp.replace(path)


def capture_case(engine,tok,r,t,root,cfg,calibration=False):
 file=cache_path(root,r,t);mp=file.with_suffix('.json')
 if mp.exists():
  meta=json.loads(mp.read_text())
  frozen=Path(root)/'frozen_cache'/key(r)/f't{t}.pt'
  if meta.get('status')!='measured' or (file.exists() and (calibration or not cfg['save_frozen_cache'] or frozen.exists())):return meta
 try:meta=prepare(tok,r,t)
 except (ValueError,KeyError,TypeError) as ex:
  meta=dict(id=str(r['id']),group_id=str(r['group_id']),relation=r.get('relation'),split=r['split'],template=t,status='excluded',reason=str(ex));dump(mp,meta);return meta
 ids=torch.tensor(meta['input_ids'],device=engine.device)
 # Failed calibration prompts need no expensive neuron arrays; their rows can
 # still be recaptured later as routing controls.
 with torch.no_grad():z=engine.model(ids[None],attention_mask=torch.ones_like(ids[None]),use_cache=False).logits[0,-1]
 g=meta['g'];top=int(z.argmax());others=z.clone();others[g]=-torch.inf;c=int(others.argmax())
 text=None;correct=False
 if top==g:text,correct=generate(engine,tok,ids,meta['aliases'],cfg)
 meta.update(c=c,top1=top==g,correct=bool(top==g and correct),generated=text,margin=float(z[g]-z[c]),target_rank=int((z>z[g]).sum())+1,
  endpoint='Canonical gold first token AND alias-prefix generation; alternative first-token aliases are not counted as successful donors')
 if calibration and not meta['correct']:
  meta['status']='calibration_not_success';dump(mp,meta);return meta
 C=engine.capture(ids,tol=cfg['tolerance']);features=extract(engine,C,meta['source'],g,c,cfg['tolerance'])
 ledger=engine.ledger(C,meta['source'],[g,c],[],cfg['tolerance'])
 meta.update(status='measured',certificate=C['certificate'],factor_certificate=float(features['factor_certificate'].max()),
  origin_margin={name:ledger[str(g)]['origins'][name]-ledger[str(c)]['origins'][name] for name in ledger[str(g)]['origins']})
 npz(file,**features)
 if cfg['save_frozen_cache'] and not calibration:save_frozen(C,Path(root)/'frozen_cache'/key(r)/f't{t}.pt')
 dump(mp,meta);return meta
