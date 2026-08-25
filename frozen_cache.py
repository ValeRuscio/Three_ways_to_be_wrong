"""Frozen-coefficient (sheaf) forward pass for Llama/Qwen2.5-style decoders.

For one prompt, caches the realized multiplicative coefficient fields
(attention probabilities A, gated-MLP gate factors g, inverse-RMS scales)
plus all residual-stream states, then re-implements each block as the frozen
*linear* map they induce.

Certificate C1: the manual (unfrozen) forward reproduces the model's logits.
Certificate C2: the frozen stack applied to the realized embeddings
reproduces every residual state and the logits (this is the contract).

Assumes: RMSNorm, RoPE (full), gated SiLU MLP, optional QKV biases (Qwen2.5),
GQA, untied or tied unembedding, no logit softcapping, no QK-norm
(Qwen3/Gemma-2 are outside this contract, as in the paper's capability matrix).
Run everything in fp32.
"""
from dataclasses import dataclass
from typing import List, Optional
import torch


# ----------------------------- primitives -----------------------------------

def rms_inv(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def norm_inv(x: torch.Tensor, eps: float, center: bool) -> torch.Tensor:
    """Frozen inverse scale; LayerNorm families center first (OLMo)."""
    xc = x - x.mean(-1, keepdim=True) if center else x
    return torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps)


def apply_norm(x: torch.Tensor, inv: torch.Tensor, w: torch.Tensor,
               center: bool) -> torch.Tensor:
    """Normalization with FROZEN inv scale.  Centering is linear, so it stays
    dynamic; only the multiplicative scale is frozen (the paper's contract:
    'with centering for LayerNorm families')."""
    xc = x - x.mean(-1, keepdim=True) if center else x
    return xc * inv * w


def rope_cos_sin(T: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device,
                                             dtype=torch.float32) / head_dim))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                      # [T, dh/2]
    emb = torch.cat((freqs, freqs), dim=-1)               # [T, dh]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rot_half(t):
    t1, t2 = t.chunk(2, dim=-1)
    return torch.cat((-t2, t1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [H, T, dh]; cos,sin: [T, dh]
    return q * cos + rot_half(q) * sin, k * cos + rot_half(k) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [Hkv, T, dh] -> [Hkv*n_rep, T, dh]
    if n_rep == 1:
        return x
    Hkv, T, dh = x.shape
    return x[:, None].expand(Hkv, n_rep, T, dh).reshape(Hkv * n_rep, T, dh)


# ----------------------------- caches ---------------------------------------

@dataclass
class LayerCache:
    A: torch.Tensor        # [H, T, T] frozen post-softmax attention probs
    gate: torch.Tensor     # [T, d_ff] frozen gate factors act(gate_proj(x_hat))
    inv1: torch.Tensor     # [T, 1]  frozen 1/rms before attention
    inv2: torch.Tensor     # [T, 1]  frozen 1/rms before MLP
    vmask: torch.Tensor = None   # [T, Hkv*dh] frozen clip active-set (OLMo)
    vconst: torch.Tensor = None  # [T, Hkv*dh] realized clipped values off-set


@dataclass
class ForwardCache:
    resid: List[torch.Tensor]   # resid[l] = residual stream entering layer l;
                                # resid[0] = embeddings, resid[L] = final state
    layers: List[LayerCache]
    inv_f: torch.Tensor         # [T, 1] frozen final-norm 1/rms
    logits: torch.Tensor        # [T, V]
    input_ids: torch.Tensor


class Weights:
    """Thin fp32 view of the model weights the frozen map needs."""

    def __init__(self, model):
        cfg = model.config
        self.L = cfg.num_hidden_layers
        self.H = cfg.num_attention_heads
        self.Hkv = getattr(cfg, "num_key_value_heads", self.H)
        self.dh = getattr(cfg, "head_dim",
                          cfg.hidden_size // cfg.num_attention_heads)
        self.n_rep = self.H // self.Hkv
        self.eps = getattr(cfg, "rms_norm_eps",
                           getattr(cfg, "layer_norm_eps", 1e-5))
        self.theta = getattr(cfg, "rope_theta", 10000.0)
        self.clip_qkv = getattr(cfg, "clip_qkv", None)     # OLMo
        dec = model.model
        # LayerNorm families (OLMo) center; RMSNorm families don't.
        self.center = "LayerNorm" in type(dec.norm).__name__
        d = cfg.hidden_size
        dev = next(model.parameters()).device
        f32 = lambda p: p.detach().float()
        ln_w = lambda mod: (f32(mod.weight)
                            if getattr(mod, "weight", None) is not None
                            else torch.ones(d, device=dev))  # OLMo: no-param LN
        self.layers = []
        for blk in dec.layers:
            a, m = blk.self_attn, blk.mlp
            self.layers.append(dict(
                ln1=ln_w(blk.input_layernorm),
                ln2=ln_w(blk.post_attention_layernorm),
                Wq=f32(a.q_proj.weight), bq=None if a.q_proj.bias is None else f32(a.q_proj.bias),
                Wk=f32(a.k_proj.weight), bk=None if a.k_proj.bias is None else f32(a.k_proj.bias),
                Wv=f32(a.v_proj.weight), bv=None if a.v_proj.bias is None else f32(a.v_proj.bias),
                Wo=f32(a.o_proj.weight),
                Wg=f32(m.gate_proj.weight), Wu=f32(m.up_proj.weight),
                Wd=f32(m.down_proj.weight),
            ))
        self.ln_f = ln_w(dec.norm)
        self.WU = f32(model.get_output_embeddings().weight)   # [V, d]
        self.WE = f32(dec.embed_tokens.weight)
        # The model's own rotary module handles rope_scaling variants
        # (llama3 long-context scaling in Llama-3.1/3.2, etc.).
        self.rotary = getattr(dec, "rotary_emb", None) or \
            dec.layers[0].self_attn.rotary_emb

    def cos_sin(self, T: int, device, dtype):
        """RoPE tables from the model's own rotary module (scaling-aware)."""
        pos = torch.arange(T, device=device).view(1, -1)
        dummy = torch.zeros(1, T, self.dh, device=device, dtype=torch.float32)
        try:
            cos, sin = self.rotary(dummy, pos)
            return cos[0].float(), sin[0].float()
        except TypeError:            # very old transformers: manual fallback
            return rope_cos_sin(T, self.dh, self.theta, device, dtype)


# ----------------------------- cache construction ---------------------------

@torch.no_grad()
def build_cache(model, W: Weights, input_ids: torch.Tensor,
                cert_tol: float = 1e-4) -> ForwardCache:
    """Manual fp32 forward; caches coefficients and states; certifies vs model."""
    dev = input_ids.device
    T = input_ids.shape[-1]
    x = W.WE[input_ids.view(-1)]                              # [T, d]
    cos, sin = W.cos_sin(T, dev, x.dtype)
    mask = torch.full((T, T), float("-inf"), device=dev).triu(1)

    resid, layers = [x.clone()], []
    for lw in W.layers:
        inv1 = norm_inv(x, W.eps, W.center)
        xn = apply_norm(x, inv1, lw["ln1"], W.center)
        q = xn @ lw["Wq"].T + (lw["bq"] if lw["bq"] is not None else 0)
        k = xn @ lw["Wk"].T + (lw["bk"] if lw["bk"] is not None else 0)
        v = xn @ lw["Wv"].T + (lw["bv"] if lw["bv"] is not None else 0)
        vmask = vconst = None
        if W.clip_qkv is not None:                            # OLMo clamp
            c = W.clip_qkv
            vmask = (v.abs() < c).float()
            vconst = v.clamp(-c, c) * (1 - vmask)             # realized const
            q, k = q.clamp(-c, c), k.clamp(-c, c)
            v = v.clamp(-c, c)
        q = q.view(T, W.H, W.dh).transpose(0, 1)
        k = repeat_kv(k.view(T, W.Hkv, W.dh).transpose(0, 1), W.n_rep)
        v = repeat_kv(v.view(T, W.Hkv, W.dh).transpose(0, 1), W.n_rep)
        q, k = apply_rope(q, k, cos, sin)
        scores = q @ k.transpose(-1, -2) / W.dh ** 0.5 + mask
        A = scores.softmax(-1)                                # [H, T, T]
        attn = (A @ v).transpose(0, 1).reshape(T, -1) @ lw["Wo"].T
        x = x + attn
        inv2 = norm_inv(x, W.eps, W.center)
        xn2 = apply_norm(x, inv2, lw["ln2"], W.center)
        gate = torch.nn.functional.silu(xn2 @ lw["Wg"].T)     # frozen field
        x = x + (gate * (xn2 @ lw["Wu"].T)) @ lw["Wd"].T
        resid.append(x.clone())
        layers.append(LayerCache(A=A, gate=gate, inv1=inv1, inv2=inv2,
                                 vmask=vmask, vconst=vconst))

    inv_f = norm_inv(x, W.eps, W.center)
    logits = apply_norm(x, inv_f, W.ln_f, W.center) @ W.WU.T

    # Certificate C1: match the model's own forward.
    ref = model(input_ids.view(1, -1)).logits[0].float()
    rel = (logits - ref).norm() / ref.norm()
    assert rel < cert_tol, f"C1 failed: manual forward rel err {rel:.2e}"
    return ForwardCache(resid=resid, layers=layers, inv_f=inv_f,
                        logits=logits, input_ids=input_ids)


# ----------------------------- frozen linear map ----------------------------

def frozen_layer(x: torch.Tensor, W: Weights, l: int, C: ForwardCache,
                 attn_edit=None) -> torch.Tensor:
    """One decoder block as the frozen LINEAR map in x (coefficients constant).

    attn_edit: optional fn(A)->A applied to the frozen probs (for edge edits
    inside the frozen account; the real-model edits live in repairs.py).
    """
    lw, lc = W.layers[l], C.layers[l]
    T = x.shape[0]
    xn = apply_norm(x, lc.inv1, lw["ln1"], W.center)          # frozen scale
    v = xn @ lw["Wv"].T + (lw["bv"] if lw["bv"] is not None else 0)
    if lc.vmask is not None:                                  # frozen clip set
        v = v * lc.vmask + lc.vconst
    v = repeat_kv(v.view(T, W.Hkv, W.dh).transpose(0, 1), W.n_rep)
    A = lc.A if attn_edit is None else attn_edit(lc.A)
    attn = (A @ v).transpose(0, 1).reshape(T, -1) @ lw["Wo"].T
    x = x + attn
    xn2 = apply_norm(x, lc.inv2, lw["ln2"], W.center)         # frozen scale
    x = x + (lc.gate * (xn2 @ lw["Wu"].T)) @ lw["Wd"].T       # frozen gate
    return x


def frozen_forward(x0: torch.Tensor, W: Weights, C: ForwardCache,
                   return_states: bool = False):
    x, states = x0, [x0]
    for l in range(W.L):
        x = frozen_layer(x, W, l, C)
        if return_states:
            states.append(x)
    logits = apply_norm(x, C.inv_f, W.ln_f, W.center) @ W.WU.T
    return (logits, states) if return_states else logits


@torch.no_grad()
def certify_frozen(W: Weights, C: ForwardCache, tol: float = 1e-4) -> float:
    """Certificate C2: frozen stack reproduces the realized pass exactly."""
    logits, states = frozen_forward(C.resid[0], W, C, return_states=True)
    worst = max((states[l] - C.resid[l]).norm() / (C.resid[l].norm() + 1e-30)
                for l in range(len(states)))
    worst = max(worst, (logits - C.logits).norm() / C.logits.norm())
    assert worst < tol, f"C2 failed: frozen reconstruction rel err {worst:.2e}"
    return float(worst)


# ----------------------------- pullback & tDLA ------------------------------

def target_delivery(xL: torch.Tensor, W: Weights, C: ForwardCache,
                    tok: int) -> torch.Tensor:
    """<u_tok, x_hat_{L,T}> with the final norm frozen -- linear in xL."""
    xh = apply_norm(xL[-1:], C.inv_f[-1:], W.ln_f, W.center)[0]
    return (xh * W.WU[tok]).sum()


@torch.enable_grad()
def tdla_edge_scores(W: Weights, C: ForwardCache, tok_g: int,
                     src_positions, qpos: int = -1,
                     tok_c: int = None) -> torch.Tensor:
    """Exact pulled-back readout paired with per-edge attention messages.

    Returns scores [L, H, |src|]: contribution of edge (l, h, j -> qpos) to the
    target logit under the frozen contract, with the cosection pulled back
    through all downstream frozen maps by (linear) autograd.
    """
    x0 = C.resid[0].clone()
    T = x0.shape[0]
    qp = qpos % T
    attn_outs = []
    x = x0
    for l in range(W.L):                       # frozen forward, retain messages
        lw, lc = W.layers[l], C.layers[l]
        xn = apply_norm(x, lc.inv1, lw["ln1"], W.center)
        v = xn @ lw["Wv"].T + (lw["bv"] if lw["bv"] is not None else 0)
        if lc.vmask is not None:
            v = v * lc.vmask + lc.vconst
        v = repeat_kv(v.view(T, W.Hkv, W.dh).transpose(0, 1), W.n_rep)
        per_edge = lc.A[:, qp, :, None] * v    # [H, T_src, dh]
        per_edge.requires_grad_(True)
        per_edge.retain_grad()
        attn_outs.append((per_edge, lw))
        attn_full = (lc.A @ v).transpose(0, 1).reshape(T, -1) @ lw["Wo"].T
        # route qpos row through the leaf so grads flow:
        Wo_h = lw["Wo"].T.view(W.H, W.dh, -1)
        row = torch.einsum("hjd,hde->e", per_edge, Wo_h)
        attn_full = attn_full.clone()
        attn_full[qp] = attn_full[qp].detach() - attn_full[qp].detach() + row
        x = x + attn_full
        xn2 = apply_norm(x, lc.inv2, lw["ln2"], W.center)
        x = x + (lc.gate * (xn2 @ lw["Wu"].T)) @ lw["Wd"].T
    xh = apply_norm(x[qp:qp + 1], C.inv_f[-1:], W.ln_f, W.center)[0]
    u = W.WU[tok_g] if tok_c is None else W.WU[tok_g] - W.WU[tok_c]
    zg = (xh * u).sum()          # target logit, or margin if tok_c given
    zg.backward()
    src = torch.as_tensor(list(src_positions))
    out = torch.stack([pe.grad[:, src, :].mul(pe[:, src, :].detach()).sum(-1)
                       for pe, _ in attn_outs])              # [L, H, |src|]
    return out.detach()
