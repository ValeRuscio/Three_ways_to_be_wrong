"""Frozen-coefficient account for GPT-NeoX / Pythia (paper Sec. 2, Table 3
'parallel block formula').

Differences from the Llama path, all absorbed into the affine maps:
  - parallel residual: x' = x + Attn(LN1(x)) + MLP(LN2(x)), both branches
    reading the SAME input state;
  - LayerNorm with weight AND bias (centering stays dynamic-linear; the
    realized inverse std is the frozen scale; biases go to the beta terms);
  - fused QKV projection with per-head [q|k|v] layout;
  - partial rotary (rotary_pct of each head dimension);
  - non-gated GELU MLP, frozen via the ratio trick:
        act(z) = g0 * z with g0 = act(z0)/z0 at the realized z0
    (g0 -> act'(0) = 1/2 as z0 -> 0), so the frozen MLP is linear in x and
    exact at the realized input.

Certificates: C1 manual forward == model logits; C2 frozen stack reproduces
all states and logits. Same tolerances as the Llama path (fp32, eager).
"""
from dataclasses import dataclass
from typing import List
import torch

from frozen_cache import rope_cos_sin, rot_half
from torch.nn.functional import linear as _lin


# ----------------------------- weights ---------------------------------------

class NeoXWeights:
    arch = "neox"
    center = True

    def __init__(self, model):
        cfg = model.config
        self.L = cfg.num_hidden_layers
        self.H = cfg.num_attention_heads
        self.dh = cfg.hidden_size // cfg.num_attention_heads
        self.rot = int(self.dh * cfg.rotary_pct)
        self.eps = cfg.layer_norm_eps
        self.theta = getattr(cfg, "rotary_emb_base",
                             getattr(cfg, "rope_theta", 10000.0))
        self.parallel = cfg.use_parallel_residual
        dec = model.gpt_neox
        f32 = lambda p: None if p is None else p.detach().float()
        self.layers = []
        for blk in dec.layers:
            a, m = blk.attention, blk.mlp
            self.layers.append(dict(
                ln1w=f32(blk.input_layernorm.weight),
                ln1b=f32(blk.input_layernorm.bias),
                ln2w=f32(blk.post_attention_layernorm.weight),
                ln2b=f32(blk.post_attention_layernorm.bias),
                Wqkv=f32(a.query_key_value.weight),
                bqkv=f32(a.query_key_value.bias),
                Wo=f32(a.dense.weight), bo=f32(a.dense.bias),
                W1=f32(m.dense_h_to_4h.weight), b1=f32(m.dense_h_to_4h.bias),
                W2=f32(m.dense_4h_to_h.weight), b2=f32(m.dense_4h_to_h.bias)))
        self.lnfw = f32(dec.final_layer_norm.weight)
        self.lnfb = f32(dec.final_layer_norm.bias)
        self.WU = f32(model.get_output_embeddings().weight)
        self.WE = f32(dec.embed_in.weight)


# ----------------------------- primitives ------------------------------------

def ln_stats(x, eps):
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    inv = torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps)
    return inv


def ln_apply(x, inv, w, b):
    """LayerNorm with FROZEN inverse std; centering stays dynamic-linear."""
    return (x - x.mean(-1, keepdim=True)) * inv * w + b


def _qkv(x_norm, lw, H, dh):
    T = x_norm.shape[0]
    qkv = _lin(x_norm, lw["Wqkv"], lw["bqkv"]).view(T, H, 3 * dh)
    q = qkv[..., :dh].transpose(0, 1)
    k = qkv[..., dh:2 * dh].transpose(0, 1)
    v = qkv[..., 2 * dh:].transpose(0, 1)
    return q, k, v


def _partial_rope(q, k, cos, sin, rot):
    qr, qp = q[..., :rot], q[..., rot:]
    kr, kp = k[..., :rot], k[..., rot:]
    qr2 = qr * cos + rot_half(qr) * sin
    kr2 = kr * cos + rot_half(kr) * sin
    return torch.cat([qr2, qp], -1), torch.cat([kr2, kp], -1)


def _gelu_ratio(z, eps=1e-6):
    g = torch.nn.functional.gelu(z)
    return torch.where(z.abs() > eps, g / z, torch.full_like(z, 0.5))


# ----------------------------- cache -----------------------------------------

@dataclass
class NeoXLayerCache:
    A: torch.Tensor       # [H, T, T]
    gate: torch.Tensor    # [T, 4d]  frozen act(z0)/z0 ratios
    inv1: torch.Tensor
    inv2: torch.Tensor


@dataclass
class NeoXCache:
    resid: List[torch.Tensor]
    layers: List[NeoXLayerCache]
    inv_f: torch.Tensor
    logits: torch.Tensor
    input_ids: torch.Tensor


@torch.no_grad()
def build_cache_neox(model, W: NeoXWeights, input_ids, cert_tol=1e-4):
    dev = input_ids.device
    T = input_ids.shape[-1]
    x = W.WE[input_ids.view(-1)]
    cos, sin = rope_cos_sin(T, W.rot, W.theta, dev, x.dtype)
    mask = torch.full((T, T), float("-inf"), device=dev).triu(1)
    resid, layers = [x.clone()], []
    for lw in W.layers:
        inv1 = ln_stats(x, W.eps)
        xa = ln_apply(x, inv1, lw["ln1w"], lw["ln1b"])
        q, k, v = _qkv(xa, lw, W.H, W.dh)
        q, k = _partial_rope(q, k, cos, sin, W.rot)
        # baddbmm mirrors the HF kernel exactly (matmul can take a
        # different fastmath path on some builds)
        scores = torch.baddbmm(
            torch.zeros(W.H, T, T, device=x.device, dtype=x.dtype),
            q, k.transpose(-1, -2), beta=0.0, alpha=W.dh ** -0.5)
        A = (scores + mask).softmax(-1)
        attn = _lin((A @ v).transpose(0, 1).reshape(T, -1),
                    lw["Wo"], lw["bo"])
        inv2 = ln_stats(x if W.parallel else x + attn, W.eps)
        xm_in = x if W.parallel else x + attn
        xm = ln_apply(xm_in, inv2, lw["ln2w"], lw["ln2b"])
        z = _lin(xm, lw["W1"], lw["b1"])
        gate = _gelu_ratio(z)
        mlp = _lin(gate * z, lw["W2"], lw["b2"])
        x = x + attn + mlp
        resid.append(x.clone())
        layers.append(NeoXLayerCache(A=A, gate=gate, inv1=inv1, inv2=inv2))
    inv_f = ln_stats(x, W.eps)
    logits = _lin(ln_apply(x, inv_f, W.lnfw, W.lnfb), W.WU)
    ref = model(input_ids.view(1, -1), use_cache=False).logits[0].float()
    rel = (logits - ref).norm() / ref.norm()
    assert rel < cert_tol, f"NeoX C1 failed: rel err {rel:.2e}"
    return NeoXCache(resid=resid, layers=layers, inv_f=inv_f,
                     logits=logits, input_ids=input_ids)


def frozen_layer_neox(x, W: NeoXWeights, l: int, C: NeoXCache):
    """One block as the frozen LINEAR map (coefficients A, gate, inv fixed)."""
    lw, lc = W.layers[l], C.layers[l]
    T = x.shape[0]
    xa = ln_apply(x, lc.inv1, lw["ln1w"], lw["ln1b"])
    _, _, v = _qkv(xa, lw, W.H, W.dh)
    attn = _lin((lc.A @ v).transpose(0, 1).reshape(T, -1),
                lw["Wo"], lw["bo"])
    xm_in = x if W.parallel else x + attn
    xm = ln_apply(xm_in, lc.inv2, lw["ln2w"], lw["ln2b"])
    z = _lin(xm, lw["W1"], lw["b1"])
    mlp = _lin(lc.gate * z, lw["W2"], lw["b2"])
    return x + attn + mlp


@torch.no_grad()
def certify_frozen_neox(W: NeoXWeights, C: NeoXCache, tol=1e-4):
    x = C.resid[0]
    worst = 0.0
    for l in range(W.L):
        x = frozen_layer_neox(x, W, l, C)
        worst = max(worst, float((x - C.resid[l + 1]).norm()
                                 / (C.resid[l + 1].norm() + 1e-30)))
    logits = _lin(ln_apply(x, C.inv_f, W.lnfw, W.lnfb), W.WU)
    worst = max(worst, float((logits - C.logits).norm() / C.logits.norm()))
    assert worst < tol, f"NeoX C2 failed: rel err {worst:.2e}"
    return worst


# ----------------------------- diagnostics -----------------------------------

def final_state(W: NeoXWeights, C: NeoXCache):
    """Realized normalized final state x_hat at the last position, [d]."""
    return ln_apply(C.resid[-1][-1:], C.inv_f[-1:], W.lnfw, W.lnfb)[0]


def target_delivery_neox(W: NeoXWeights, C: NeoXCache, tok: int):
    return float((final_state(W, C) * W.WU[tok]).sum())


@torch.enable_grad()
def tdla_neox(W: NeoXWeights, C: NeoXCache, tok_g, S, tok_c=None):
    """Pulled-back readout paired with per-edge source messages, [L,H,|S|].
    Parallel-block version: the MLP branch reads the pre-attention state."""
    x = C.resid[0].clone()
    T = x.shape[0]
    leaves = []
    for l in range(W.L):
        lw, lc = W.layers[l], C.layers[l]
        xa = ln_apply(x, lc.inv1, lw["ln1w"], lw["ln1b"])
        _, _, v = _qkv(xa, lw, W.H, W.dh)
        per_edge = lc.A[:, -1, :, None] * v          # [H, T_src, dh]
        per_edge.requires_grad_(True)
        per_edge.retain_grad()
        leaves.append(per_edge)
        attn = _lin((lc.A @ v).transpose(0, 1).reshape(T, -1),
                lw["Wo"], lw["bo"])
        Wo_h = lw["Wo"].T.view(W.H, W.dh, -1)
        row = torch.einsum("hjd,hde->e", per_edge, Wo_h) + lw["bo"]
        attn = attn.clone()
        attn[-1] = attn[-1].detach() - attn[-1].detach() + row
        xm_in = x if W.parallel else x + attn
        xm = ln_apply(xm_in, lc.inv2, lw["ln2w"], lw["ln2b"])
        z = _lin(xm, lw["W1"], lw["b1"])
        x = x + attn + _lin(lc.gate * z, lw["W2"], lw["b2"])
    xh = ln_apply(x[-1:], C.inv_f[-1:], W.lnfw, W.lnfb)[0]
    u = W.WU[tok_g] if tok_c is None else W.WU[tok_g] - W.WU[tok_c]
    (xh * u).sum().backward()
    src = torch.as_tensor(list(S))
    return torch.stack([(pe.grad[:, src] * pe[:, src].detach()).sum(-1)
                        for pe in leaves]).detach()
