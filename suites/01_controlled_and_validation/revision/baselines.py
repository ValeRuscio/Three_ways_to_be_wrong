"""Matched scalar edge-gate interventions, with endpoint gradients and path IG.

These are explicit gate baselines, NOT an implementation of AtP* or published EAP-IG.
Each candidate gates selected attention probabilities and renormalizes the row.
All attribution and measured effects use that same intervention.
"""
import time
import numpy as np
import torch
from repairs import attention_edits


def gate_edit(columns, candidate, alpha, base_edit=None):
    def edit(A, li):
        A = base_edit(A,li) if base_edit is not None else A
        if candidate is not None and li!=candidate[0]: return A
        H,T,_=A.shape
        mask=torch.zeros_like(A)
        heads=list(range(H)) if candidate is None else [candidate[1]]
        for h in heads: mask[h,-1,list(columns)]=1.
        B=A*(1.-mask+mask*alpha)
        return B/B.sum(-1,keepdim=True).clamp_min(1e-12)
    return edit


def margin_at(model,W,ids,g,c,columns,candidate,alpha,base_edit=None,gradient=False):
    with torch.set_grad_enabled(gradient):
        a=torch.tensor(float(alpha),device=ids.device,requires_grad=gradient)
        with attention_edits(model,W,prob_edit=gate_edit(columns,candidate,a,base_edit)):
            z=model(ids[None],use_cache=False).logits[0,-1]
            m=z[g]-z[c]
        if gradient: return float(torch.autograd.grad(m,a)[0])
        return float(m)


def matched_scores(model,W,ids,g,c,columns,candidate=None,steps=16,base_edit=None):
    if steps<2: raise ValueError('IG requires at least 2 intervals')
    start=time.perf_counter()
    m1=margin_at(model,W,ids,g,c,columns,candidate,1,base_edit)
    m0=margin_at(model,W,ids,g,c,columns,candidate,0,base_edit)
    # Nested grid: coarse and fine estimates without repeated gradient computations.
    grads=np.array([margin_at(model,W,ids,g,c,columns,candidate,a,base_edit,True)
                    for a in np.linspace(0,1,2*steps+1)])
    fine=float(np.trapezoid(grads,dx=1/(2*steps)))
    coarse=float(np.trapezoid(grads[::2],dx=1/steps))
    h=1e-3
    fd=(margin_at(model,W,ids,g,c,columns,candidate,1+h,base_edit)-
        margin_at(model,W,ids,g,c,columns,candidate,1-h,base_edit))/(2*h)
    return dict(gate_gradient=float(grads[-1]),gate_ig=fine,gate_ig_coarse=coarse,
        exact_drop=m1-m0,ig_residual=fine-(m1-m0),ig_refinement=abs(fine-coarse),
        gradient_fd_error=float(grads[-1]-fd),seconds=time.perf_counter()-start)


def identity_certificate(model,W,ids,base_edit=None,tol=1e-4):
    with torch.no_grad():
        if base_edit:
            with attention_edits(model,W,prob_edit=base_edit): ref=model(ids[None],use_cache=False).logits
        else: ref=model(ids[None],use_cache=False).logits
        with attention_edits(model,W,prob_edit=gate_edit([0],None,1.,base_edit)):
            actual=model(ids[None],use_cache=False).logits
        err=float((ref-actual).norm()/ref.norm().clamp_min(1e-12))
    if err>=tol: raise AssertionError(f'Live gate identity failed: {err}')
    return err


def legacy_gradxact(model,W,C,ids,g,c,source_sets,base_edit=None):
    """Original head-context gradient x source-message score; separate from gate gradient.
    This predicts unrenormalized message removal, unlike the renormalized gate oracle.
    Native projection-input hooks retain the model's attention implementation.
    Returns [number of source sets, L, H].
    """
    from contextlib import nullcontext
    from ranking import _source_messages
    contexts={}; handles=[]
    try:
        for l,block in enumerate(model.model.layers):
            def hook(module,args,l=l):
                x=args[0]
                if not x.requires_grad: x=x.detach().requires_grad_(True)
                contexts[l]=x
                return (x,*args[1:])
            handles.append(block.self_attn.o_proj.register_forward_pre_hook(hook))
        with torch.enable_grad():
            with attention_edits(model,W,prob_edit=base_edit) if base_edit else nullcontext():
                z=model(ids[None],use_cache=False).logits[0,-1]
            derivatives=torch.autograd.grad(z[g]-z[c],[contexts[l] for l in range(W.L)])
        grads=torch.stack([v.reshape(-1,W.H,W.dh)[-1] for v in derivatives])
        with torch.no_grad():
            return torch.stack([(grads*_source_messages(W,C,S)[0]).sum(-1) for S in source_sets]).detach()
    finally:
        for h in handles: h.remove()
