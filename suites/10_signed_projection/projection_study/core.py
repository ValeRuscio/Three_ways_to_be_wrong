import torch
from channel_study.core import live
from followup_study.mechanisms import freezes,MECHANISMS
from .helpers import residual_gradients,cosine

CONDITIONS={'live':(), 'attention_only':('attention',), 'gates_only':('gate',), 'scales_only':('norm',), 'all_frozen':MECHANISMS}


def projections(gradients,mean):
 mean=mean.double();norm=mean.norm(dim=-1);unit=mean/norm.clamp_min(1e-30)[:,None]
 rows=[];paired=[]
 for l in range(len(mean)):
  valid=norm[l]>1e-15;G={k:v[l].double() for k,v in gradients.items()}
  p={k:float(v@unit[l]) if valid else None for k,v in G.items()};den=float(G['all_frozen'].norm())
  total=p['live']-p['all_frozen'] if valid else None
  def scale(x):return x/den if x is not None and den>1e-15 else None
  for condition,g in G.items():
   effect=p['live']-p[condition] if valid else None
   rows.append(dict(layer=l,condition=condition,projection=p[condition],cosine=cosine(g,mean[l]),gradient_norm=float(g.norm()),reference_mean_norm=float(norm[l]),
    removed_projection=effect,projection_over_content_norm=scale(p[condition]),removed_over_content_norm=scale(effect),
    remaining_vs_all=(p[condition]-p['all_frozen']) if valid else None,total_selection_projection=total,
    content_gradient_norm=den))
  e={k:p['live']-v for k,v in p.items()} if valid else {k:None for k in p}
  contrasts=dict(norm_minus_gate=(e['scales_only']-e['gates_only']) if valid else None,
   norm_minus_attention=(e['scales_only']-e['attention_only']) if valid else None,
   singleton_sum_minus_total=(e['scales_only']+e['gates_only']+e['attention_only']-total) if valid else None)
  paired.append(dict(layer=l,**contrasts,**{k+'_over_content_norm':scale(v) for k,v in contrasts.items()}))
 return rows,paired


def measure(e,C,g,c,mean,baseline_atol=1e-4,baseline_rtol=1e-5):
 gradients={};values={}
 for condition,mask in CONDITIONS.items():gradients[condition],values[condition]=residual_gradients(e,C,g,c,mask)
 native=float(C['z'][g]-C['z'][c]);err=max(abs(v-native) for v in values.values())
 if err>baseline_atol+baseline_rtol*abs(native):raise AssertionError(f'Baseline changed under freeze: {err}')
 rows,paired=projections(gradients,mean)
 return dict(rows=rows,paired=paired,objective_values=values,native_margin=native,baseline_error=err,native_certificate=C['certificate']),gradients


def finite_checks(e,C,g,c,mean,gradients,cfg):
 """Intervene on x_l at the answer token; realized mechanisms held at baseline
 for selected groups. Compare directional derivative with centered differences.
 """
 layers=sorted({round(v*(e.L-1)) for v in cfg['fd_layer_fractions']});rows=[]
 for l in layers:
  mu=mean[l].double();mn=float(mu.norm())
  if mn<=1e-15:continue
  direction=(mu/mn).to(e.device,dtype=torch.float32)
  xn=float(C['layers'][l]['observed_x'][-1].norm())
  for condition,mask in CONDITIONS.items():
   derivative=float(gradients[condition][l].double()@(mu/mn))
   for epsilon in cfg['fd_epsilons']:
    alpha=epsilon*max(xn,1e-12);vals=[]
    for sign in [-1,1]:
     def edit(mod,args,kw):
      x=(args[0] if args else kw['hidden_states']).clone();x[0,-1]+=sign*alpha*direction
      if args:return (x,*args[1:]),kw
      return args,dict(kw,hidden_states=x)
     with freezes(e,C,mask):
      h=e.blocks[l].register_forward_pre_hook(edit,with_kwargs=True)
      try:
       with torch.no_grad():z=live(e,C['ids']);vals.append(float(z[g]-z[c]))
      finally:h.remove()
    observed=vals[1]-vals[0];predicted=2*alpha*derivative;err=abs(observed-predicted)
    rows.append(dict(layer=l,condition=condition,epsilon=epsilon,delta_norm=alpha,projection=derivative,predicted_difference=predicted,
     measured_difference=observed,absolute_error=err,passed=err<=cfg['fd_atol']+cfg['fd_rtol']*max(abs(observed),abs(predicted)),resolved=abs(predicted)>10*cfg['fd_atol']))
 return rows
