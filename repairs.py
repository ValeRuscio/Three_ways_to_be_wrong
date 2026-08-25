"""Verdict-specific causal repairs, applied to the LIVE model (not the frozen
account), so recovery is a genuine causal claim.

Three intervention families, one per verdict class:

  presence  -> source-state patch: residual states over the source span, in a
               layer band, are replaced by donor states from a matched success
               (same fact under a template/context where the model succeeds).
  transport -> edge repair: attention probs on the top-k tDLA source->output
               edges are boosted (row renormalized) at the final position.
  selection -> demoter ablation: the top-k heads with the most negative exact
               ledger contribution to the margin u = u_g - u_c have their
               output zeroed at the final position.

Controls: each repair applied to every verdict class (the off-diagonal of the
repair matrix), plus random-edge / random-head / random-band versions.

Implementation: attention edits need access to post-softmax probs, so the
relevant LlamaAttention/Qwen2Attention forwards are swapped for an eager
reimplementation reusing the module's own weights.  Certificate C-live: with
no edit installed, the wrapped model reproduces the unwrapped logits.
"""
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import torch

from frozen_cache import (Weights, ForwardCache, rope_cos_sin, apply_rope,
                          repeat_kv, tdla_edge_scores)


# ----------------------- wrapped eager attention -----------------------------

def _eager_attn_forward(module, hidden, W: Weights, layer_idx: int,
                        prob_edit: Optional[Callable], head_zero: Sequence[int]):
    T = hidden.shape[1]
    x = hidden[0]
    dev, dt = x.device, x.dtype
    q = module.q_proj(x); k = module.k_proj(x); v = module.v_proj(x)
    if W.clip_qkv is not None:                                # OLMo
        q, k, v = (t.clamp(-W.clip_qkv, W.clip_qkv) for t in (q, k, v))
    q = q.view(T, W.H, W.dh).transpose(0, 1)
    k = repeat_kv(k.view(T, W.Hkv, W.dh).transpose(0, 1), W.n_rep)
    v = repeat_kv(v.view(T, W.Hkv, W.dh).transpose(0, 1), W.n_rep)
    cos, sin = W.cos_sin(T, dev, dt)
    cos, sin = cos.to(dt), sin.to(dt)
    q, k = apply_rope(q, k, cos, sin)
    mask = torch.full((T, T), torch.finfo(dt).min, device=dev, dtype=dt).triu(1)
    A = ((q @ k.transpose(-1, -2)) / W.dh ** 0.5 + mask).softmax(-1)
    if prob_edit is not None:
        A = prob_edit(A, layer_idx)
    out = A @ v                                   # [H, T, dh]
    for h in head_zero:
        out[h, -1, :] = 0.0                       # ablate head at final position
    out = out.transpose(0, 1).reshape(T, -1)
    return module.o_proj(out).unsqueeze(0)


@contextmanager
def attention_edits(model, W: Weights,
                    prob_edit: Optional[Callable] = None,
                    head_zero: Dict[int, List[int]] = None):
    """prob_edit(A, layer_idx)->A on post-softmax probs; head_zero {layer:[h]}."""
    head_zero = head_zero or {}
    layers = model.model.layers
    saved = []
    try:
        for li, blk in enumerate(layers):
            if prob_edit is None and li not in head_zero:
                continue
            attn = blk.self_attn
            saved.append((attn, attn.forward))
            def mk(attn=attn, li=li):
                def fwd(hidden_states, *a, **kw):
                    out = _eager_attn_forward(attn, hidden_states, W, li,
                                              prob_edit, head_zero.get(li, []))
                    # transformers < 4.48 expects a 3-tuple
                    # (attn_out, attn_weights, past_key_value); newer, a 2-tuple.
                    import transformers
                    major, minor = map(int,
                                       transformers.__version__.split(".")[:2])
                    if (major, minor) < (4, 48):
                        return out, None, kw.get("past_key_value")
                    return out, None
                return fwd
            attn.forward = mk()
        yield
    finally:
        for attn, f in saved:
            attn.forward = f


@contextmanager
def resid_patch(model, layer_band: Sequence[int], positions: Sequence[int],
                donor_states: Dict[int, torch.Tensor], alpha: float = 1.0):
    """Replace residual-stream slices at the *input* of each layer in band.

    donor_states[l] : [T_d, d] donor residual stream entering layer l; donor
    positions must be pre-aligned to the receiver's `positions` (same length).
    """
    hooks = []
    pos = torch.as_tensor(list(positions))
    try:
        for l in layer_band:
            def hook(module, args, kwargs, l=l):
                hs = args[0]
                hs = hs.clone()
                hs[0, pos] = (1 - alpha) * hs[0, pos] + \
                    alpha * donor_states[l][:len(pos)].to(hs.dtype)
                return (hs, *args[1:]), kwargs
            hooks.append(model.model.layers[l].register_forward_pre_hook(
                hook, with_kwargs=True))
        yield
    finally:
        for h in hooks:
            h.remove()


# ----------------------------- measurements ---------------------------------

@torch.no_grad()
def margin(model, input_ids, tok_g: int, tok_c: int) -> Tuple[float, bool]:
    # use_cache=False: wrapped attentions skip KV-cache updates, which would
    # break unwrapped downstream layers when only a subset is edited.
    z = model(input_ids.view(1, -1), use_cache=False).logits[0, -1].float()
    return float(z[tok_g] - z[tok_c]), bool(z.argmax() == tok_g)


# ----------------------------- repair library --------------------------------

@torch.no_grad()
def repair_presence(model, input_ids, tok_g, tok_c, S, donor_states,
                    layer_band, alpha=1.0):
    with resid_patch(model, layer_band, S, donor_states, alpha):
        return margin(model, input_ids, tok_g, tok_c)


@torch.no_grad()
def repair_transport(model, W: Weights, C: ForwardCache, input_ids,
                     tok_g, tok_c, S, k=8, boost=None, target_row_mass=0.5):
    """Boost top-k tDLA edges at the final position; renormalize rows."""
    scores = tdla_edge_scores(W, C, tok_g, S)               # [L, H, |S|]
    flat = scores.flatten()
    top = torch.topk(flat, k).indices
    Lh, Hh, Sn = scores.shape
    edges: Dict[int, List[Tuple[int, int]]] = {}
    for idx in top.tolist():
        l = idx // (Hh * Sn); r = idx % (Hh * Sn)
        edges.setdefault(l, []).append((r // Sn, list(S)[r % Sn]))

    def prob_edit(A, li):
        if li not in edges:
            return A
        A = A.clone()
        for h, j in edges[li]:
            if boost is not None:
                A[h, -1, j] = A[h, -1, j] * boost
            else:                                   # set row mass on the edge
                A[h, -1, j] = target_row_mass
        A[:, -1, :] = A[:, -1, :] / A[:, -1, :].sum(-1, keepdim=True)
        return A

    with attention_edits(model, W, prob_edit=prob_edit):
        return margin(model, input_ids, tok_g, tok_c)


@torch.no_grad()
def repair_selection(model, W: Weights, C: ForwardCache, input_ids,
                     tok_g, tok_c, k=4):
    """Ablate the k heads whose exact ledger contribution to u = u_g - u_c is
    most negative at the final position (the demoters)."""
    u_hat = ((C.inv_f[-1] * W.ln_f) * (W.WU[tok_g] - W.WU[tok_c])).squeeze()
    from frozen_cache import apply_norm
    contribs = []
    for l in range(W.L):
        lw, lc = W.layers[l], C.layers[l]
        x = C.resid[l]
        xn = apply_norm(x, lc.inv1, lw["ln1"], W.center)
        v = xn @ lw["Wv"].T + (lw["bv"] if lw["bv"] is not None else 0)
        if lc.vmask is not None:
            v = v * lc.vmask + lc.vconst
        v = repeat_kv(v.view(x.shape[0], W.Hkv, W.dh).transpose(0, 1), W.n_rep)
        head_out = lc.A[:, -1, :, None].mul(v).sum(1)       # [H, dh]
        Wo_h = lw["Wo"].T.view(W.H, W.dh, -1)
        proj = torch.einsum("hd,hde->he", head_out, Wo_h)   # [H, d]
        c_l = proj @ u_hat                                  # [H]
        if W.center:   # centering is linear: subtract mean-component share
            c_l = c_l - proj.mean(-1) * u_hat.sum()
        contribs.append(c_l)
    contribs = torch.stack(contribs)                        # [L, H]
    flat = contribs.flatten()
    worst = torch.topk(-flat, k).indices
    head_zero: Dict[int, List[int]] = {}
    for idx in worst.tolist():
        head_zero.setdefault(idx // W.H, []).append(idx % W.H)
    with attention_edits(model, W, head_zero=head_zero):
        return margin(model, input_ids, tok_g, tok_c)


@torch.no_grad()
def repair_transport_force(model, W: Weights, C: ForwardCache, input_ids,
                           tok_g, tok_c, S, k=8, alpha=1.0):
    """Strong transport repair: force-attention at the mover edges.

    For the top-k tDLA edges (l, h, j), the head's final-position attention
    row is alpha-mixed toward one-hot on the source token j, so the head
    actually CARRIES the source value message instead of having its
    probability nudged (the weak arm's failure mode: renormalized boosts
    inject almost nothing against base margins of -3..-8).

    Verdict-specific by construction: if the source stalk holds no
    target-bearing section (presence failure), the forced message carries
    nothing useful; if delivery is already at success level (selection),
    forcing changes little. Only transport failures should respond strongly.
    """
    scores = tdla_edge_scores(W, C, tok_g, S)               # [L, H, |S|]
    flat = scores.flatten()
    top = torch.topk(flat, min(k, flat.numel())).indices
    Lh, Hh, Sn = scores.shape
    edges: Dict[int, List[Tuple[int, int]]] = {}
    for idx in top.tolist():
        l = idx // (Hh * Sn); r = idx % (Hh * Sn)
        edges.setdefault(l, []).append((r // Sn, list(S)[r % Sn]))

    def prob_edit(A, li):
        if li not in edges:
            return A
        A = A.clone()
        for h, j in edges[li]:
            row = torch.zeros_like(A[h, -1])
            row[j] = 1.0
            A[h, -1] = (1 - alpha) * A[h, -1] + alpha * row
        return A

    with attention_edits(model, W, prob_edit=prob_edit):
        return margin(model, input_ids, tok_g, tok_c)


@torch.no_grad()
def transport_dose_response(model, W, C, input_ids, tok_g, tok_c, S,
                            ks=(4, 8, 16), alphas=(0.25, 0.5, 1.0)):
    """Margin change grid over intervention strength -- report effects as
    dose-response, not only flips (base margins are often < -3)."""
    m0, _ = margin(model, input_ids, tok_g, tok_c)
    out = {}
    for k in ks:
        for a in alphas:
            m, fl = repair_transport_force(model, W, C, input_ids, tok_g,
                                           tok_c, S, k=k, alpha=a)
            out[(k, a)] = (m - m0, fl)
    return m0, out


# ----------------------------- controls --------------------------------------

@torch.no_grad()
def repair_random_heads(model, W, input_ids, tok_g, tok_c, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, W.L * W.H, (k,), generator=g)
    hz: Dict[int, List[int]] = {}
    for i in idx.tolist():
        hz.setdefault(i // W.H, []).append(i % W.H)
    with attention_edits(model, W, head_zero=hz):
        return margin(model, input_ids, tok_g, tok_c)


@torch.no_grad()
def certify_wrappers(model, W, input_ids, tol=1e-3):
    """C-live: wrapped attention with identity edit == original model."""
    base = model(input_ids.view(1, -1), use_cache=False).logits[0, -1].float()
    with attention_edits(model, W, prob_edit=lambda A, li: A):
        z = model(input_ids.view(1, -1), use_cache=False).logits[0, -1].float()
    rel = (z - base).norm() / base.norm()
    assert rel < tol, f"C-live failed: wrapper rel err {rel:.2e}"
    return float(rel)
