"""Live interventions with fixed pre-intervention competitor, graded margin gains,
controls, full grids, and separately reported native doses and output-state cost.
No automatic diagonal-dominance assertion or cross-unit dose ratio.
"""
from contextlib import nullcontext,ExitStack
import random
import torch
from .engine import residual_patch,attention_scale,live,norm
from .data import render,span,target_tokens,matches,digest,random_span
from .measure import generation,run_q


def scaled(delta,norm_value):
 n=float(delta.norm())
 return delta*(norm_value/n) if n>1e-12 else None


def donors(engine,tok,r,facts,cfg,chat_tok=None):
 donor=None;attempts=[]
 for ti in [1,2]:
  prompt=render(r,ti,chat_tok);ids=torch.tensor(tok(prompt,add_special_tokens=chat_tok is None).input_ids,device=engine.device)
  try:S=span(tok,prompt,r['subject'],add_special_tokens=chat_tok is None);g=target_tokens(tok,prompt,r['answer'],chat_tok is not None)[0]
  except ValueError:continue
  if len(S)!=len(r['source']) or g!=r['g']:attempts.append(dict(template=ti,reason='unaligned_tokens'));continue
  z=live(engine,ids)
  if int(z.argmax())!=r['g']:attempts.append(dict(template=ti,reason='wrong_first_token'));continue
  text=generation(engine,tok,ids,cfg['max_new_tokens'])
  if not matches(text,r['aliases']):attempts.append(dict(template=ti,reason='wrong_full_answer'));continue
  donor=(engine.capture(ids),S);attempts.append(dict(template=ti,reason='verified',generated=text));break
 wrong=None
 # Independent calibration donors only; no evaluation outcomes used to tune donor search.
 candidates=[f for f in facts if f['split']=='calibration' and f.get('eligible') and f['correct'] and f['relation']==r['relation'] and f['group_id']!=r['group_id'] and f['g']!=r['g'] and len(f['source'])==len(r['source'])]
 candidates.sort(key=lambda f:digest([r['id'],f['id']]))
 if candidates:
  f=candidates[0];wrong=(engine.capture(torch.tensor(f['input_ids'],device=engine.device)),f['source'])
 return donor,wrong,attempts


def execute_grid(engine,r,C,cfg,donor=None,wrong=None,plant=None):
 ids=C['ids'];g=r['g'];c=r['c'];S=r['source'];layers=sorted(set(min(engine.L-1,max(0,int(f*engine.L))) for f in cfg['layer_fractions']))
 generator=torch.Generator(device='cpu').manual_seed(int(digest([cfg['seed'],r['id'],plant])[:8],16))
 def noise(shape):return torch.randn(shape,generator=generator).to(engine.device)
 base_z=C['z'].clone()
 if plant and plant['kind']=='selection':base_z[c]+=plant['bias']
 base_margin=float(base_z[g]-base_z[c]);base_last=C['layers'][-1]['next'][-1]
 rows=[]
 source_states=engine.component(C,S);competitor_pullbacks=engine.pullbacks(C,c)
 donor_states=engine.component(donor[0],donor[1]) if donor else None
 def measure(family,arm,dose,layer=None,ctx=None,output_delta=None):
  # Caller keeps planted modification installed. Non-output interventions rerun live network.
  last={}
  def hook(mod,args):last['x']=args[0][0,-1].detach()
  h=engine.final.register_forward_pre_hook(hook)
  try:
   with ctx if ctx is not None else nullcontext():z=live(engine,ids).clone()
  finally:h.remove()
  if plant and plant['kind']=='selection':z[c]+=plant['bias']
  last_cost=float((last['x']-base_last).norm()/base_last.norm().clamp_min(1e-9))
  normalized_base=norm(base_last,engine.final);normalized_after=norm(last['x'],engine.final)
  if output_delta is not None:normalized_after=normalized_after+output_delta
  normalized_cost=float((normalized_after-normalized_base).norm()/normalized_base.norm().clamp_min(1e-9))
  if output_delta is not None:
   z+=engine.head.weight@output_delta
   # Output edit acts after normalization; keep this separate from pre-norm cost.
   last_cost=None
  rows.append(dict(id=r['id'],group_id=r['group_id'],relation=r['relation'],label=r['labels'].get('full','undetermined'),
    correct_before=r['correct'],family=family,arm=arm,dose=float(dose),layer=layer,
    margin_gain=float(z[g]-z[c])-base_margin,recovered=int(z.argmax())==g,
    final_residual_relative_cost=last_cost,final_normalized_relative_cost=normalized_cost,output_edit_norm=float(output_delta.norm()) if output_delta is not None else None,
    plant=plant['kind'] if plant else 'natural'))
 for l in layers:
  if donor:
   D,DS=donor;delta=D['layers'][l]['x'][DS]-C['layers'][l]['observed_x'][S];size=float(delta.norm())
   rand=scaled(noise(delta.shape),size)
   for a in cfg['alphas']:
    measure('source','donor',a,l,residual_patch(engine,l,S,delta,a))
    if rand is not None:measure('source','random',a,l,residual_patch(engine,l,S,rand,a))
    if wrong:
     WC,WS=wrong;wd=scaled(WC['layers'][l]['x'][WS]-C['layers'][l]['x'][S],size)
     if wd is not None:measure('source','wrong_subject',a,l,residual_patch(engine,l,S,wd,a))
  if donor:
   D,DS=donor;component=donor_states[l][-1:]
   # Secondary: add donor input-S provenance at the answer position.
   random_component=scaled(noise(component.shape),float(component.norm()))
   for a in cfg['alphas']:
    measure('transport_secondary','donor_provenance',a,l,residual_patch(engine,l,[len(ids)-1],component,a))
    if random_component is not None:measure('transport_secondary','random',a,l,residual_patch(engine,l,[len(ids)-1],random_component,a))
  # Exploratory upstream selection edit: project out along LOCAL pulled-back c,
  # not raw unembedding u_c, which lives in a different representation space.
  source=source_states[l][S];rc=competitor_pullbacks[l][S]
  delta=-(source*rc).sum(-1,keepdim=True)*rc/rc.square().sum(-1,keepdim=True).clamp_min(1e-12)
  rand=scaled(noise(delta.shape),float(delta.norm()))
  for a in cfg['alphas']:
   measure('selection_upstream','competitor_projection',a,l,residual_patch(engine,l,S,delta,a))
   if rand is not None:measure('selection_upstream','random',a,l,residual_patch(engine,l,S,rand,a))
 scores=engine.head_scores(C,S,g);indices=scores.flatten().argsort(descending=True);k=min(cfg['top_heads'],scores.numel())
 top=[(int(i)//engine.H,int(i)%engine.H) for i in indices[:k]];bottom=[(int(i)//engine.H,int(i)%engine.H) for i in indices[-k:]]
 R=random_span(len(ids),S,random.Random(int(digest(r['id'])[:8],16)))
 for gamma in cfg['gammas']:
  measure('transport','top_heads',gamma,ctx=attention_scale(engine,S,top,gamma))
  measure('transport','bottom_heads',gamma,ctx=attention_scale(engine,S,bottom,gamma))
  if R is not None:measure('transport','random_span',gamma,ctx=attention_scale(engine,R,top,gamma))
 # All output edits include all-vocabulary effects, not just a target logit overwrite.
 ug=engine.head.weight[g];uc=engine.head.weight[c];eg=ug/ug.norm();ec=uc/uc.norm();rv=noise(ug.shape);rv/=rv.norm()
 if 'median_proj_g' in r:
  h=norm(base_last,engine.final);up=max(0.,r['median_proj_g']-float(h@eg));down=max(0.,float(h@ec)-r['median_proj_c'])
  for alpha in cfg['alphas']:
   measure('selection_calibrated','answer_up',alpha,output_delta=alpha*up*eg)
   measure('selection_calibrated','alternative_down',alpha,output_delta=-alpha*down*ec)
   measure('selection_calibrated','random_up_norm',alpha,output_delta=alpha*up*rv)
   measure('selection_calibrated','random_down_norm',alpha,output_delta=alpha*down*rv)
 for d in cfg['selection_doses']:
  measure('selection','answer_up',d,output_delta=d*eg)
  measure('selection','alternative_down',d,output_delta=-d*ec)
  measure('selection','random',d,output_delta=d*rv)
 return rows


def planted(engine,r,C,cfg):
 """Paired known modifications + exact reversal control. Output-only plant has
 no hidden-state changes. Exact edge cuts cannot be undone by multiplicative
 boosting; this is explicitly marked, not interpreted as failed diagnosis.
 """
 S=r['source'];g=r['g'];c=r['c'];ids=C['ids'];layer=max(0,engine.L//2)
 gen=torch.Generator().manual_seed(int(digest([r['id'],'plant'])[:8],16))
 x=C['layers'][layer]['x'][S];delta=torch.randn(x.shape,generator=gen).to(x);delta=scaled(delta,float(x.norm()))
 heads=[(l,h) for l in range(engine.L) for h in range(engine.H)]
 cases=[('source',residual_patch(engine,layer,S,delta,1.)),('transport',attention_scale(engine,S,heads,0.)),('selection',nullcontext())]
 output=[];audit=[]
 for kind,ctx in cases:
  try:
   with ctx:
    P=engine.capture(ids);spec={'kind':kind}
    if kind=='selection':spec['bias']=float(C['z'][g]-C['z'][c])+1.
    z=P['z'].clone()
    if kind=='selection':z[c]+=spec['bias']
    failed=int(z.argmax())!=g
    rec=dict(r);rec['correct']=not failed
    ledger=engine.ledger(P,S,[g],[]);tau=ledger[str(g)]['origins']['subject'];share=tau/max(abs(float(z[g])),1e-6)
    lens=engine.lens(P,S,[g]+r['pool_tokens']);q=run_q(lens[:,:,0]-lens[:,:,1:].amax(-1),cfg['consecutive'])
    label='correct' if not failed else 'undetermined' if 'tau_threshold' not in r else 'source' if q<=0 else 'transport' if tau<r['tau_threshold'] or share<r['share_threshold'] else 'selection'
    rec['labels']={'full':label};audit.append(dict(id=r['id'],group_id=r['group_id'],operation=kind,failed=failed,label=label,
      exact_reversal_recovered=True,original_margin=float(C['z'][g]-C['z'][c]),plant_margin=float(z[g]-z[c]),
      boost_cannot_restore_zero_edges=kind=='transport',status='completed'))
    # Keep ineffective interventions in audit, but do not pretend they are failures.
    if failed:output.extend(execute_grid(engine,rec,P,cfg,donor=(C,S),plant=spec))
  except RuntimeError as e:
   if 'removes all available mass' not in str(e):raise
   audit.append(dict(id=r['id'],group_id=r['group_id'],operation=kind,status='undefined_zero_attention_row'))
 # Verify removal of temporary interventions really restores native logits.
 if not torch.allclose(live(engine,ids),C['z'],atol=1e-4,rtol=1e-4):raise AssertionError('Plant did not reverse')
 return output,audit
