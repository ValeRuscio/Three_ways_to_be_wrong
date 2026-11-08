import random
import numpy as np
import torch
from .engine import norm
from .data import digest,span,target_tokens,random_span,matches,render


def run_q(lens,k):
 # lens: positions x layers (gold minus strongest decoy)
 if lens.shape[1]<k:return float('-inf')
 return max(float(lens[:,i:i+k].min(1).values.max()) for i in range(lens.shape[1]-k+1))


@torch.no_grad()
def generation(engine,tok,ids,n):
 result=engine.model.generate(ids[None],max_new_tokens=n,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
 return tok.decode(result[0,len(ids):],skip_special_tokens=True)


def measure(engine,tok,row,prompt,answers,hard_tokens,cfg,chat=False):
 r={k:row[k] for k in ['id','group_id','subject','relation','answer','aliases','split']};r['prompt']=prompt
 r['popularity']=row.get('popularity');ids=torch.tensor(tok(prompt,add_special_tokens=not chat).input_ids,device=engine.device)
 r['input_ids']=ids.tolist();r['generated']=generation(engine,tok,ids,cfg['max_new_tokens']);r['correct']=matches(r['generated'],row['aliases'])
 try:
  S=span(tok,prompt,row['subject'],add_special_tokens=not chat);g=target_tokens(tok,prompt,row['answer'],chat)[0]
  alias=set(target_tokens(tok,prompt,a,chat)[0] for a in row['aliases'])
 except ValueError as e:r.update(eligible=False,exclusion=str(e));return r,None
 C=engine.capture(ids);z=C['z'];top=int(z.argmax());c=int(z.topk(2).indices[1]) if top==g else top
 h=norm(C['layers'][-1]['next'][-1],engine.final)
 r['proj_g']=float(h@engine.head.weight[g]/engine.head.weight[g].norm());r['proj_c']=float(h@engine.head.weight[c]/engine.head.weight[c].norm())
 r.update(g=g,c=c,source=S,top=top,first_token_correct=top in alias,canonical_first_correct=top==g,margin=float(z[g]-z[c]),gold_logit=float(z[g]),competitor_logit=float(z[c]),certificate=C['certificate'])
 if r['correct'] and top!=g:reason='correct_alternative_initial_token'
 elif not r['correct'] and top==g:reason='later_token_failure'
 elif not r['correct'] and top in alias:reason='ambiguous_alias_initial_token'
 elif any(normalized_in(a,prompt) for a in row['aliases']):reason='answer_copied_in_prompt'
 else:reason=None
 if abs(float(z[g]))<1e-6 or abs(float(z[c]))<1e-6:reason=reason or 'near_zero_logit_share_undefined'
 pool=set()
 for answer in answers:
  try:pool.add(target_tokens(tok,prompt,answer,chat)[0])
  except ValueError:pass
 pool=sorted(pool-alias)
 if len(pool)<20:reason=reason or 'fewer_than_20_decoys'
 r.update(eligible=reason is None,exclusion=reason,pool_tokens=pool)
 if not pool:return r,C
 lens=engine.lens(C,S,[g]+pool);strength=lens[:,:,1:].amax((0,1)).cpu().numpy()
 hard=sorted([pool[i] for i in np.argsort(strength)[-20:]]) if hard_tokens is None else hard_tokens
 if not set(hard).issubset(pool):raise ValueError('Shared hard decoy tokens incompatible with paired tokenizer/prompt')
 r['hard_tokens']=hard;r['q']={};r['q_max']={}
 for convention,selected in [('full',pool),('hard20',hard)]:
  cols=[pool.index(t)+1 for t in selected];margins=lens[:,:,0]-lens[:,:,cols].amax(-1)
  r['q'][convention]=run_q(margins,cfg['consecutive']);r['q_max'][convention]=float(margins.max())
 rng=random.Random(int(digest([cfg['seed'],row['id']])[:12],16));R=random_span(len(ids),S,rng)
 r['random_span']=R
 decoy=rng.choice([token for token in pool if token!=c]);r['replaced_competitor']=decoy
 # Exact accounting for all eligible records, including calibration and successes.
 if r['eligible']:
  ledger=engine.ledger(C,S,list(dict.fromkeys([g,c,decoy])),R or [])
  r['ledger']=ledger;r['tau']=ledger[str(g)]['origins']['subject'];r['share']=r['tau']/max(abs(float(z[g])),1e-6)
  cc=ledger[str(c)]['origins'];r['competitor_share']=cc['subject']/max(abs(float(z[c])),1e-6)
  r['competitor_absolute_share']=abs(cc['subject'])/max(sum(abs(v) for v in cc.values()),1e-6)
 return r,C


def normalized_in(answer,prompt):
 from .data import normalize
 a=normalize(answer);p=normalize(prompt)
 return bool(a) and (' '+a+' ') in (' '+p+' ')


def assign_labels(records,cfg,training=False):
 successful=[r for r in records if r.get('eligible') and r['split']=='calibration' and r['correct']]
 # Balanced relation cycling, then a declared global cap; never evaluation successes.
 by={rel:sorted([r for r in successful if r['relation']==rel],key=lambda r:digest(r['id'])) for rel in sorted({r['relation'] for r in successful})}
 cal=[rs[i] for i in range(max(map(len,by.values()),default=0)) for rs in by.values() if i<len(rs)][:cfg['calibration_reference_n']]
 reference_ids={r['id'] for r in cal}
 for r in records:
  r['calibration_reference']=r['id'] in reference_ids
  r['labels']={};r['calibration_n']=0;r['calibration_mode']='unavailable'
  if not r.get('eligible'):continue
  refs=cal;mode='pooled'
  if training:
   same=[a for a in successful if a['relation']==r['relation']]
   refs=same[:cfg['calibration_reference_n']];mode='within_relation'
   if len(refs)<cfg['min_calibration']:
    refs=[a for a in cal if a['relation']!=r['relation']];mode='leave_relation_out'
  r['calibration_n']=len(refs);r['calibration_mode']=mode
  enough=len(refs)>=cfg['min_calibration']
  if enough:
   tt=float(np.quantile([a['tau'] for a in refs],cfg['quantile']));ss=float(np.quantile([a['share'] for a in refs],cfg['quantile']))
   floor=float(np.quantile([a['tau'] for a in refs],cfg['handoff_quantile']));r.update(tau_threshold=tt,share_threshold=ss,handoff_floor=floor)
   if all('proj_g' in a for a in refs):
    r['median_proj_g']=float(np.median([a['proj_g'] for a in refs]));r['median_proj_c']=float(np.median([a['proj_c'] for a in refs]))
  for convention in ['full','hard20']:
   r['labels'][convention]='correct' if r['correct'] else 'undetermined' if not enough else 'source' if r['q'][convention]<=0 else 'transport' if r['tau']<tt or r['share']<ss else 'selection'
  final=r['tau'];curve=r['ledger'][str(r['g'])]['handoff']
  r['handoff_cut']=next((i for i,v in enumerate(curve) if v>=.5*final),None) if enough and final>max(0,floor) else None
 return records
