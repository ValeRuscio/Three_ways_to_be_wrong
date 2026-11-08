"""Native-pass affine ledger. No Jacobian approximation: freeze realized A, gates,
and inverse norm scales. Biases and LayerNorm centering remain explicit.
Pinned to transformers 4.48.3; one unpadded sequence, fp32, eager, no KV cache.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def norm(x, module, inv=None):
    center=isinstance(module,torch.nn.LayerNorm)
    y=x-x.mean(-1,keepdim=True) if center else x
    eps=getattr(module,'variance_epsilon',getattr(module,'eps',1e-5))
    if inv is None: inv=torch.rsqrt(y.float().square().mean(-1,keepdim=True)+eps)
    y=y*inv
    if getattr(module,'weight',None) is not None: y=y*module.weight
    if getattr(module,'bias',None) is not None: y=y+module.bias
    return y


class Engine:
    def __init__(self,model):
        self.model=model.eval(); self.kind=model.config.model_type
        if self.kind not in ('llama','qwen2','gpt_neox','olmo2'):
            raise ValueError(f'Unsupported architecture: {self.kind}')
        for p in model.parameters(): p.requires_grad_(False)
        if next(model.parameters()).dtype!=torch.float32: raise ValueError('Certificate requires fp32')
        cfg=model.config
        if getattr(cfg,'use_sliding_window',False): raise ValueError('Sliding window unsupported')
        self.dec=model.gpt_neox if self.kind=='gpt_neox' else model.model
        self.blocks=self.dec.layers; self.L=len(self.blocks); self.H=cfg.num_attention_heads
        self.Hkv=getattr(cfg,'num_key_value_heads',self.H); self.dh=cfg.hidden_size//self.H
        self.final=self.dec.final_layer_norm if self.kind=='gpt_neox' else self.dec.norm
        self.head=model.get_output_embeddings()
        self.device=next(model.parameters()).device

    @torch.no_grad()
    def capture(self,ids,tol=1e-4):
        records=[{} for _ in self.blocks]; handles=[]
        def save_input(dst,key):
            def hook(module,args,kw):
                x=args[0] if args else kw['hidden_states']; dst[key]=x[0].detach().clone()
            return hook
        def save_norm(dst,key):
            def hook(module,args):
                x=args[0][0]; y=x-x.mean(-1,keepdim=True) if isinstance(module,torch.nn.LayerNorm) else x
                eps=getattr(module,'variance_epsilon',getattr(module,'eps',1e-5))
                dst[key]=torch.rsqrt(y.square().mean(-1,keepdim=True)+eps).detach()
            return hook
        for l,b in enumerate(self.blocks):
            d=records[l]
            handles.append(b.register_forward_pre_hook(save_input(d,'x'),with_kwargs=True))
            def save_out(module,args,out,d=d): d['next']=out[0][0].detach().clone()
            handles.append(b.register_forward_hook(save_out))
            names=['input_layernorm','post_attention_layernorm'] if self.kind!='olmo2' else ['post_attention_layernorm','post_feedforward_layernorm']
            for name in names: handles.append(getattr(b,name).register_forward_pre_hook(save_norm(d,name)))
            proj=b.mlp.dense_h_to_4h if self.kind=='gpt_neox' else b.mlp.gate_proj
            def save_gate(module,args,out,d=d,b=b):
                z=out[0].detach()
                if self.kind=='gpt_neox':
                    act=b.mlp.act(z); d['gate']=torch.where(z.abs()>1e-7,act/z,torch.full_like(z,.5))
                else: d['gate']=b.mlp.act_fn(z)
            handles.append(proj.register_forward_hook(save_gate))
        final={}; handles.append(self.final.register_forward_pre_hook(save_norm(final,'inv')))
        try: result=self.model(ids[None],use_cache=False,output_attentions=True)
        finally:
            for h in handles:h.remove()
        for l,(d,a) in enumerate(zip(records,result.attentions)):
            d['A']=a[0].detach();d['observed_x']=d['x']
            if l>0:
                d['input_offset']=d['x']-records[l-1]['next'];d['x']=records[l-1]['next']
        C=dict(layers=records,inv=final['inv'],z=result.logits[0,-1].detach(),ids=ids.detach())
        worst=0.
        for l,d in enumerate(records):
            x=self.layer(d['x'],l,C)
            worst=max(worst,float((x-d['next']).norm()/d['next'].norm().clamp_min(1)))
        z=self.read(records[-1]['next'],C)
        worst=max(worst,float((z-C['z']).norm()/C['z'].norm().clamp_min(1)))
        if worst>tol: raise AssertionError(f'Frozen reconstruction error {worst:g} > {tol}')
        C['certificate']=worst
        return C

    def values(self,x,l):
        b=self.blocks[l]
        if self.kind=='gpt_neox':
            v=b.attention.query_key_value(x).view(len(x),self.H,3*self.dh)[...,2*self.dh:]
        else:v=b.self_attn.v_proj(x).view(len(x),self.Hkv,self.dh)
        v=v.transpose(0,1)
        return v.repeat_interleave(self.H//v.shape[0],dim=0)

    def layer(self,x,l,C,A_override=None):
        b=self.blocks[l]; d=C['layers'][l]; A=d['A'] if A_override is None else A_override
        x=x+d.get('input_offset',0)
        if self.kind=='olmo2':
            att=b.self_attn.o_proj((A@self.values(x,l)).transpose(0,1).reshape(len(x),-1))
            y=x+norm(att,b.post_attention_layernorm,d['post_attention_layernorm'])
            mlp=b.mlp.down_proj(d['gate']*b.mlp.up_proj(y))
            return y+norm(mlp,b.post_feedforward_layernorm,d['post_feedforward_layernorm'])
        a=norm(x,b.input_layernorm,d['input_layernorm'])
        proj=b.attention.dense if self.kind=='gpt_neox' else b.self_attn.o_proj
        att=proj((A@self.values(a,l)).transpose(0,1).reshape(len(x),-1))
        y=x+att
        mi=x if self.kind=='gpt_neox' and self.model.config.use_parallel_residual else y
        mi=norm(mi,b.post_attention_layernorm,d['post_attention_layernorm'])
        if self.kind=='gpt_neox': mlp=b.mlp.dense_4h_to_h(d['gate']*b.mlp.dense_h_to_4h(mi))
        else: mlp=b.mlp.down_proj(d['gate']*b.mlp.up_proj(mi))
        return y+mlp

    def linear(self,x,l,C): return self.layer(x,l,C)-self.layer(torch.zeros_like(x),l,C)

    def read(self,x,C,tokens=None):
        h=norm(x[-1],self.final,C['inv'][-1])
        if tokens is None:return self.head(h)
        bias=self.head.bias[tokens] if self.head.bias is not None else None
        return F.linear(h,self.head.weight[tokens],bias)

    @torch.no_grad()
    def lens(self,C,S,tokens):
        return torch.stack([norm(d['next'][S],self.final)@self.head.weight[tokens].T for d in C['layers']],dim=1)

    def pullbacks(self,C,token):
        x=C['layers'][-1]['next'].detach().requires_grad_(True)
        with torch.enable_grad(): r=torch.autograd.grad(self.read(x,C,[token]).sum(),x)[0].detach()
        rs=[r]
        for l in reversed(range(self.L)):
            x=C['layers'][l]['x'].detach().requires_grad_(True)
            with torch.enable_grad(): r=torch.autograd.grad(self.layer(x,l,C),x,grad_outputs=r)[0].detach()
            rs.append(r)
        return list(reversed(rs))

    @torch.no_grad()
    def component(self,C,S):
        x=torch.zeros_like(C['layers'][0]['x']); x[S]=C['layers'][0]['x'][S]
        states=[x]
        for l in range(self.L): x=self.linear(x,l,C);states.append(x)
        return states

    def ledger(self,C,S,tokens,random_span,tol=1e-4):
        """Disjoint input origins: subject takes precedence over first position.
        Bias includes all layer affine offsets, final norm bias, unembedding bias.
        """
        x0=C['layers'][0]['x']; source=self.component(C,S); out={}; positions=list(range(len(x0)))
        groups={'subject':list(S),'first_token':[] if 0 in S else [0],
                'template':[i for i in positions if i not in S and i!=0]}
        for token in tokens:
            rs=self.pullbacks(C,token); per=(rs[0]*x0).sum(-1)
            contributions={k:float(per[v].sum()) for k,v in groups.items()}
            biases=[]
            with torch.no_grad():
                for l in range(self.L): biases.append(float((rs[l+1]*self.layer(torch.zeros_like(x0),l,C)).sum()))
                end=float(self.read(torch.zeros_like(x0),C,[token])[0]); contributions['bias']=sum(biases)+end
                actual=float(C['z'][token]); error=abs(sum(contributions.values())-actual)/max(1,abs(actual))
                p=[(r*x).sum(-1) for r,x in zip(rs,source)]
                total=[float(v.sum()) for v in p]; handoff_error=max(abs(v-total[-1]) for v in total)/max(1,abs(total[-1]))
            if max(error,handoff_error)>tol: raise AssertionError(f'Origin/handoff certificate failed {error}, {handoff_error}')
            out[str(token)]=dict(origins=contributions,position_contributions=per.cpu().tolist(),layer_biases=biases,
                final_bias=end,logit=actual,reconstruction_error=error,handoff_error=handoff_error,
                handoff=[float(v[-1]) for v in p],total_by_cut=total,random_span_support=float(per[random_span].sum()))
        return out

    def head_scores(self,C,S,token):
        """Subject-input-origin messages, NOT contextual-state edge sums.
        Score the direct S-position -> T message at each head with downstream pullback,
        including the within-block frozen MLP or OLMo post-attention normalization.
        """
        rs=self.pullbacks(C,token); states=self.component(C,S); scores=[]
        for l in range(self.L):
            a=torch.ones(self.H,device=self.device,requires_grad=True)
            mask=torch.zeros_like(C['layers'][l]['A']);mask[:,-1,S]=1
            with torch.enable_grad():
                A=C['layers'][l]['A']*(1+mask*(a[:,None,None]-1))
                y=self.layer(states[l],l,C,A)-self.layer(torch.zeros_like(states[l]),l,C,A)
                score=torch.autograd.grad(y,a,grad_outputs=rs[l+1])[0]
            scores.append(score.detach())
        return torch.stack(scores)


@contextmanager
def residual_patch(engine,layer,S,delta,alpha=1.):
    def hook(module,args,kw):
        x=args[0] if args else kw['hidden_states']; y=x.clone();y[:,S]+=alpha*delta[None]
        return ((y,*args[1:]),kw) if args else (args,dict(kw,hidden_states=y))
    h=engine.blocks[layer].register_forward_pre_hook(hook,with_kwargs=True)
    try:yield
    finally:h.remove()


@contextmanager
def attention_scale(engine,span,heads,gamma):
    """Native eager attention -> modify returned attention weights -> recompute
    its output from LIVE values. Downstream computations and norms stay live.
    Original attention still runs, making this slower but architecture-faithful.
    """
    handles=[]
    for l,b in enumerate(engine.blocks):
        selected=[h for li,h in heads if li==l]
        if not selected:continue
        module=b.attention if engine.kind=='gpt_neox' else b.self_attn
        def hook(module,args,kw,out,l=l,selected=selected,b=b):
            hidden=args[0] if args else kw['hidden_states']; x=hidden[0]
            A=out[-1] if engine.kind=='gpt_neox' else out[1]
            if A is None:raise RuntimeError('Call live model with output_attentions=True')
            A=A[0].clone()
            for h in selected:A[h,-1,span]*=gamma
            den=A.sum(-1,keepdim=True)
            if bool((den<=0).any()):raise RuntimeError('Attention intervention removes all available mass')
            A=A/den
            proj=b.attention.dense if engine.kind=='gpt_neox' else b.self_attn.o_proj
            value=proj((A@engine.values(x,l)).transpose(0,1).reshape(len(x),-1))[None]
            parts=list(out);parts[0]=value
            parts[-1 if engine.kind=='gpt_neox' else 1]=A[None]
            return tuple(parts)
        handles.append(module.register_forward_hook(hook,with_kwargs=True))
    try:yield
    finally:
        for h in handles:h.remove()


@torch.no_grad()
def live(engine,ids):return engine.model(ids[None],use_cache=False,output_attentions=True).logits[0,-1]
