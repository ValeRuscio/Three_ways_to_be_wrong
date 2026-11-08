"""Activation adjoints and finite intervention tests. FP32, unpadded eager HF.
Frozen means realized A, SiLU outputs, and inverse RMS scales held constant.
"""
from contextlib import contextmanager
import math
import numpy as np
import torch
from channel_study.core import live,modules,dot
from followup_study.mechanisms import freezes,MECHANISMS,collect


def groups(T,S):
    S=set(map(int,S)); first={0}-S; template=set(range(T))-S-first
    return {'subject':sorted(S),'first':sorted(first),'template':sorted(template),
            'subject_last':[max(S)],'answer':[T-1]}


def metric(live_g,frozen_g):
    a=live_g.double().flatten(); b=frozen_g.double().flatten(); s=a-b
    an=float(a.norm());bn=float(b.norm());sn=float(s.norm());cross=float(b@s)
    cos=lambda x,y:float(x@y/(x.norm()*y.norm())) if float(x.norm()*y.norm())>1e-20 else None
    angle=math.degrees(math.acos(max(-1.,min(1.,cos(a,b))))) if cos(a,b) is not None else None
    return dict(live_norm=an,frozen_norm=bn,selection_norm=sn,angle_degrees=angle,
        live_frozen_cosine=cos(a,b),content_selection_cosine=cos(b,s),
        frozen_over_live=bn/an if an>1e-15 else None,
        selection_over_live=sn/an if an>1e-15 else None,
        selection_fraction=sn/(sn+bn) if sn+bn>1e-15 else None,
        signed_cross=cross,closure_error=abs(an**2-bn**2-sn**2-2*cross)/max(1.,an**2))


def adjoints(e,C,g,c,which=()):
    cuts={};handles=[]
    with freezes(e,C,which):
        try:
            handles.append(e.model.get_input_embeddings().register_forward_hook(lambda m,a,o:o.detach().requires_grad_(True)))
            for l,b in enumerate(e.blocks):
                def grab(m,args,kw,l=l):cuts[l]=args[0] if args else kw['hidden_states']
                handles.append(b.register_forward_pre_hook(grab,with_kwargs=True))
            handles.append(e.final.register_forward_pre_hook(lambda m,a:cuts.__setitem__(e.L,a[0])))
            z=live(e,C['ids']);m=z[g]-z[c]
            grad=torch.autograd.grad(m,[cuts[l] for l in range(e.L+1)])
            return torch.stack([v[0].detach().cpu() for v in grad]),float(m.detach())
        finally:
            for h in handles:h.remove()


def measure_adjoints(e,C,g,c,S):
    L,lv=adjoints(e,C,g,c);F,fv=adjoints(e,C,g,c,MECHANISMS)
    native=float(C['z'][g]-C['z'][c]);err=max(abs(lv-native),abs(fv-native))
    if err>1e-4+1e-5*abs(native):raise AssertionError('Adjoint baseline changed')
    gg=groups(len(C['ids']),S);by_position=[];by_role=[]
    x0=e.model.get_input_embeddings()(C['ids'][None]).detach()
    input_terms=(F[0].double()*x0[0].cpu().double()).sum(-1)
    with torch.no_grad(),freezes(e,C,MECHANISMS),embedding_input(e,torch.zeros_like(x0)):
        bz=live(e,C['ids']);bias=float(bz[g]-bz[c])
    origin_error=abs(float(input_terms.sum())+bias-native)
    if origin_error>1e-4+1e-5*abs(native):raise AssertionError('Input origin reconstruction failed')
    for l in range(e.L+1):
        for t in range(len(C['ids'])):
            origin=next(k for k in ['subject','first','template'] if t in gg[k])
            by_position.append(dict(layer=l,position=t,origin=origin,**metric(L[l,t],F[l,t])))
        for role,idx in gg.items():
            if idx:by_role.append(dict(layer=l,role=role,positions=len(idx),frozen_origin_margin=float(input_terms[idx].sum()) if l==0 else None,**metric(L[l,idx],F[l,idx])))
    return dict(roles=by_role,positions=by_position,baseline_error=err,native_certificate=C['certificate'],input_affine_bias=bias,input_origin_error=origin_error),L,F


@contextmanager
def embedding_input(e,x):
    h=e.model.get_input_embeddings().register_forward_hook(lambda m,a,o:x)
    try:yield
    finally:h.remove()


def embedding_checks(e,C,g,c,S,L,F,cfg):
    x=e.model.get_input_embeddings()(C['ids'][None]).detach();out=[]
    for role,idx in groups(len(C['ids']),S).items():
        if role not in ['subject','first','template'] or not idx:continue
        rng=torch.Generator().manual_seed(cfg['seed']+len(idx));random=torch.randn((len(idx),x.shape[-1]),generator=rng)
        for kind,d in [('random',random),('live_aligned',L[0,idx]),('frozen_aligned',F[0,idx])]:
            dn=float(d.norm())
            if dn<1e-15:continue
            direction=torch.zeros_like(x);direction[0,idx]=(d/dn).to(x)
            for condition,mask,G in [('live',(),L),('frozen',MECHANISMS,F)]:
                slope=float((G[0].double()*direction[0].cpu().double()).sum())
                for eps in cfg['fd_epsilons']:
                    alpha=eps*max(float(x[0,idx].norm()),1e-12);vals=[]
                    with torch.no_grad(),freezes(e,C,mask):
                        for sign in [-1,1]:
                            with embedding_input(e,x+sign*alpha*direction):
                                z=live(e,C['ids']);vals.append(float(z[g]-z[c]))
                    actual=vals[1]-vals[0];pred=2*alpha*slope;error=abs(actual-pred)
                    out.append(dict(role=role,direction=kind,condition=condition,epsilon=eps,
                        actual=actual,predicted=pred,absolute_error=error,
                        resolved=abs(pred)>10*cfg['fd_atol'],
                        passed=error<=cfg['fd_atol']+cfg['fd_rtol']*max(abs(actual),abs(pred))))
    return out


def gate_jacobian(e,C,S,cfg,stop=None):
    """Rademacher v: E ||J^T v||² = ||J||F²; v is NOT unit-normalized.
    Rebuild forward for each layer to bound graph memory. Eight VJPs/layer.
    """
    x=e.model.get_input_embeddings()(C['ids'][None]).detach().requires_grad_(True)
    roles=groups(len(C['ids']),S);out=[]
    for l,b in enumerate(e.blocks):
        if stop is not None and stop.exists():return None
        gate=[];h=b.mlp.act_fn.register_forward_hook(lambda m,a,o:gate.append(o[0,-1]))
        try:
            with embedding_input(e,x):live(e,C['ids'])
        finally:h.remove()
        samples={role:[] for role in ['subject','template','first'] if roles[role]}
        rng=torch.Generator(device=e.device).manual_seed(cfg['seed']+1009*l)
        for j in range(cfg['jacobian_probes']):
            v=(torch.randint(0,2,gate[0].shape,device=e.device,generator=rng)*2-1).to(x.dtype)
            d=torch.autograd.grad(gate[0]@v,x,retain_graph=j+1<cfg['jacobian_probes'])[0][0]
            for role in samples:samples[role].append(float(d[roles[role]].double().square().sum()))
        for role,vals in samples.items():
            arr=np.asarray(vals);out.append(dict(layer=l,role=role,positions=len(roles[role]),probes=len(vals),
                jacobian_sq=float(arr.mean()),probe_se_sq=float(arr.std(ddof=1)/math.sqrt(len(arr))) if len(arr)>1 else None,
                jacobian_sq_per_position=float(arr.mean()/len(roles[role])),probe_sq=vals))
        del gate,d
        print(f'  gate Jacobian layer {l+1}/{e.L}',flush=True)
    cross={}
    for normalized in [False,True]:
        field='jacobian_sq_per_position' if normalized else 'jacobian_sq';tag='per_position' if normalized else 'total'
        values={l:{r['role']:r[field] for r in out if r['layer']==l} for l in range(e.L)}
        wins=[l for l,v in values.items() if v.get('subject',0)>v.get('template',float('inf'))]
        persistent=[l for l in range(e.L-cfg['crossing_run']+1) if all(j in wins for j in range(l,l+cfg['crossing_run']))]
        cross[tag+'_first_crossing']=min(wins) if wins else None
        cross[tag+'_persistent_crossing']=min(persistent) if persistent else None
        cross[tag+'_crossed']=bool(persistent)
        cross[tag+'_reversals']=sum((l in wins)!=(l-1 in wins) for l in range(1,e.L))
    return dict(rows=out,crossover=cross)


@contextmanager
def weight_delta(mod,direction,alpha):
    original=mod.weight.detach().cpu().clone()
    try:
        with torch.no_grad():mod.weight.copy_(original.to(mod.weight)+alpha*direction.to(mod.weight))
        yield
    finally:
        with torch.no_grad():mod.weight.copy_(original.to(mod.weight))
        if not torch.equal(mod.weight.detach().cpu(),original):raise AssertionError('Weight restoration failed')


def dense_direction(factors,mod):
    if factors is None:return None,0.
    d,x=factors;G=d.T@x;norm=float(G.double().norm())
    return (-G/norm if norm>1e-15 else None),norm


def contraction(factors,direction):
    if factors is None:return 0.
    d,x=factors
    return float(((d.double()@direction.double())*x.double()).sum())


def sgd_experiment(e,C,g,c,cfg,stop=None):
    """One selected matrix at a time, first-gold-token cross entropy.
    This is restricted SGD, not multi-matrix fine-tuning, not sequence CE.
    """
    names={n:m for n,m in modules(e).items() if int(n.split(':')[0]) in cfg['weight_layers'] and n.split(':')[1] in cfg['weight_families']}
    obj=lambda z:z[g]-z[c]
    CE,ce=collect(e,C,lambda z:-torch.log_softmax(z,dim=0)[g],selected=names)
    L,base=collect(e,C,obj,selected=names);F,fb=collect(e,C,obj,MECHANISMS,selected=names)
    out=[]
    if abs(base-fb)>1e-4+1e-5*abs(base):raise AssertionError('SGD freeze baseline changed')
    for name,mod in names.items():
        if stop is not None and stop.exists():return None
        direction,gn=dense_direction(CE.get(name),mod)
        if direction is None:continue
        lp=contraction(L.get(name),direction);fp=contraction(F.get(name),direction);wn=float(mod.weight.norm())
        for dose in cfg['step_doses']:
            alpha=dose*wn
            with weight_delta(mod,direction,alpha),torch.no_grad():
                z=live(e,C['ids']);actual=float(obj(z))-base;loss=float(-torch.log_softmax(z,dim=0)[g]);top1=int(z.argmax())==g
                with freezes(e,C,MECHANISMS):fz=live(e,C['ids']);factual=float(obj(fz))-fb
            error=abs(factual-alpha*fp);limit=cfg['fd_atol']+cfg['fd_rtol']*max(abs(factual),abs(alpha*fp))
            out.append(dict(layer=int(name.split(':')[0]),family=name.split(':')[1],dose=dose,delta_norm=alpha,sgd_learning_rate=alpha/gn,
                baseline_margin=base,baseline_ce=ce,ce_after=loss,gold_top1=top1,live_gain=actual,
                frozen_gain=factual,content_linear=alpha*fp,selection_linear=alpha*(lp-fp),
                live_linear=alpha*lp,live_nonlinear_remainder=actual-alpha*lp,
                finite_reselection=actual-factual,frozen_linear_error=error,frozen_linear_pass=error<=limit,
                baseline_gold_top1=int(C['z'].argmax())==g))
    return out


def matrix_hvp(e,C,g,c,mod,direction,mask):
    p=mod.weight;p.requires_grad_(True)
    try:
        with freezes(e,C,mask):
            z=live(e,C['ids']);m=z[g]-z[c]
            if not m.requires_grad:return 0.
            grad=torch.autograd.grad(m,p,create_graph=True,allow_unused=True)[0]
            if grad is None or not grad.requires_grad:return 0.
            along=(grad*direction.to(p)).sum()
            if not along.requires_grad:return 0.
            h=torch.autograd.grad(along,p,allow_unused=True)[0]
            return float((h*direction.to(p)).double().sum()) if h is not None else 0.
    finally:p.requires_grad_(False);p.grad=None


def curvature(e,C,g,c,cfg,stop=None):
    names={n:m for n,m in modules(e).items() if int(n.split(':')[0]) in cfg['weight_layers'] and n.split(':')[1] in cfg['weight_families']}
    L,base=collect(e,C,lambda z:z[g]-z[c],selected=names);out=[]
    for name,mod in names.items():
        if stop is not None and stop.exists():return None
        direction,gn=dense_direction(L.get(name),mod)
        if direction is None:continue
        hl=matrix_hvp(e,C,g,c,mod,direction,());hf=matrix_hvp(e,C,g,c,mod,direction,MECHANISMS)
        for eps in cfg['curvature_epsilons']:
            alpha=eps*float(mod.weight.norm());vals=[]
            for sign in [-1,1]:
                with weight_delta(mod,direction,sign*alpha),torch.no_grad():
                    z=live(e,C['ids']);vals.append(float(z[g]-z[c]))
            second=vals[0]+vals[1]-2*base;pred=alpha**2*hl
            out.append(dict(layer=int(name.split(':')[0]),family=name.split(':')[1],epsilon=eps,
                live_directional_hessian=hl,frozen_directional_hessian=hf,second_difference=second,
                predicted_second_difference=pred,absolute_error=abs(second-pred),resolved=abs(pred)>10*cfg['fd_atol'],
                passed=abs(second-pred)<=cfg['fd_atol']+cfg['fd_rtol']*max(abs(second),abs(pred))))
    return out
