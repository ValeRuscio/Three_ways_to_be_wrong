"""Blinded causal ranking: does the affine ledger predict, from one cached
forward pass, which structures will matter under subsequent interventions?

Candidates: (layer, head) source->final edge groups.  Six rankers score every
candidate from the SAME cached pass (rankings frozen before any intervention);
ground truth is the live-model margin change from ablating each candidate's
source attention mass (zero A[h, -1, S], renormalize).

Rankers:
  affine      exact pulled-back margin readout paired with the candidate's
             transported message (tDLA, margin readout)
  attention  raw attention mass on the source columns
  dla        direct-to-logit attribution of the candidate's source message
             (no pullback through downstream layers)
  gradxact   +<dm/d(head context), source part>  on the LIVE model
             (first-order / attribution-patching prediction of the ablation)
  actnorm    norm of the candidate's source message after W_O
  random     seeded noise

Metrics: Spearman(predicted, observed); precision@k vs the measured oracle;
cumulative margin drop from jointly ablating each ranker's top-k; regret@k
vs the oracle ranking (oracle = sort by measured single-candidate effects).
"""
from contextlib import contextmanager
from typing import Dict, List, Sequence
import torch

from frozen_cache import (Weights, ForwardCache, apply_norm, repeat_kv,
                          tdla_edge_scores)
from repairs import attention_edits, margin

RANKERS = ("affine", "attention", "dla", "gradxact", "actnorm", "random")


# ---------------------- per-head source messages (cached) --------------------

def _source_messages(W: Weights, C: ForwardCache, S: Sequence[int]):
    """Per (l,h): source-column context part s_h = sum_{j in S} A[h,-1,j] v_j
    (pre-W_O, [L,H,dh]) and its projection through W_O ([L,H,d])."""
    T = C.resid[0].shape[0]
    pre, post = [], []
    for l in range(W.L):
        lw, lc = W.layers[l], C.layers[l]
        xn = apply_norm(C.resid[l], lc.inv1, lw["ln1"], W.center)
        v = xn @ lw["Wv"].T + (lw["bv"] if lw["bv"] is not None else 0)
        if lc.vmask is not None:
            v = v * lc.vmask + lc.vconst
        v = repeat_kv(v.view(T, W.Hkv, W.dh).transpose(0, 1), W.n_rep)
        s = (lc.A[:, -1, S, None] * v[:, S]).sum(1)          # [H, dh]
        Wo_h = lw["Wo"].T.view(W.H, W.dh, -1)
        pre.append(s)
        post.append(torch.einsum("hd,hde->he", s, Wo_h))     # [H, d]
    return torch.stack(pre), torch.stack(post)               # [L,H,dh],[L,H,d]


# ---------------------- live capture for grad x activation -------------------

@contextmanager
def _capture_heads(model, W: Weights, store: dict):
    """Wrap attentions to expose per-head contexts [H,T,dh] as graph leaves."""
    from frozen_cache import apply_rope
    saved = []
    try:
        for li, blk in enumerate(model.model.layers):
            attn = blk.self_attn
            saved.append((attn, attn.forward))

            def mk(attn=attn, li=li):
                def fwd(hidden_states, *a, **kw):
                    x = hidden_states[0]
                    T = x.shape[0]
                    q = attn.q_proj(x); k = attn.k_proj(x); v = attn.v_proj(x)
                    if W.clip_qkv is not None:
                        q, k, v = (t.clamp(-W.clip_qkv, W.clip_qkv)
                                   for t in (q, k, v))
                    q = q.view(T, W.H, W.dh).transpose(0, 1)
                    k = repeat_kv(k.view(T, W.Hkv, W.dh).transpose(0, 1),
                                  W.n_rep)
                    v = repeat_kv(v.view(T, W.Hkv, W.dh).transpose(0, 1),
                                  W.n_rep)
                    cos, sin = W.cos_sin(T, x.device, x.dtype)
                    q, k = apply_rope(q, k, cos.to(x.dtype), sin.to(x.dtype))
                    mask = torch.full((T, T), torch.finfo(x.dtype).min,
                                      device=x.device, dtype=x.dtype).triu(1)
                    A = ((q @ k.transpose(-1, -2)) / W.dh ** 0.5
                         + mask).softmax(-1)
                    out = A @ v                              # [H, T, dh]
                    out.retain_grad()
                    store[li] = out
                    o = attn.o_proj(out.transpose(0, 1).reshape(T, -1))
                    import transformers
                    mj, mn = map(int, transformers.__version__.split(".")[:2])
                    if (mj, mn) < (4, 48):
                        return o.unsqueeze(0), None, kw.get("past_key_value")
                    return o.unsqueeze(0), None
                return fwd
            attn.forward = mk()
        yield
    finally:
        for attn, f in saved:
            attn.forward = f


def _gradxact(model, W: Weights, ids, tok_g, tok_c, s_pre):
    """+<dm/d(head context at final pos), source part>  per (l,h).

    Sign convention established by audit_gradxact.py on an analytic linear
    model where the exact ablation drop is u.W_h s_h: the first-order
    predictor of drop = m0 - m_ablated is +<grad, source part> (rho = +1.0
    on the calibration model; the minus convention gives exactly -1.0).
    Frozen before real-model evaluation."""
    store = {}
    with torch.enable_grad(), _capture_heads(model, W, store):
        z = model(ids.view(1, -1), use_cache=False).logits[0, -1]
        (z[tok_g] - z[tok_c]).backward()
    g = torch.stack([store[l].grad[:, -1] for l in range(W.L)])  # [L,H,dh]
    return (g * s_pre).sum(-1).detach()                          # [L,H]


# ---------------------- scoring + ground truth -------------------------------

@torch.no_grad()
def score_candidates(model, W: Weights, C: ForwardCache, ids, tok_g, tok_c,
                     S, seed=0) -> Dict[str, torch.Tensor]:
    """All six [L,H] rankings from one cached pass (+1 live backward)."""
    s_pre, s_post = _source_messages(W, C, S)
    u = ((C.inv_f[-1] * W.ln_f) * (W.WU[tok_g] - W.WU[tok_c])).squeeze()
    dla = s_post @ u
    if W.center:
        dla = dla - s_post.mean(-1) * u.sum()
    out = {
        "affine": tdla_edge_scores(W, C, tok_g, S, tok_c=tok_c).sum(-1),
        "attention": torch.stack([C.layers[l].A[:, -1, S].sum(-1)
                                  for l in range(W.L)]),
        "dla": dla,
        "actnorm": s_post.norm(dim=-1),
        "random": torch.randn(W.L, W.H,
                              generator=torch.Generator().manual_seed(seed)
                              ).to(dla.device),
    }
    out["gradxact"] = _gradxact(model, W, ids, tok_g, tok_c, s_pre)
    return {k: v.detach().float().cpu() for k, v in out.items()}


def _edge_edit(cands, S):
    """prob_edit zeroing A[h,-1,S] (renormalized) for candidates {(l,h)}."""
    by_layer = {}
    for l, h in cands:
        by_layer.setdefault(l, []).append(h)

    def edit(A, li):
        if li not in by_layer:
            return A
        A = A.clone()
        for h in by_layer[li]:
            A[h, -1, list(S)] = 0.0
            A[h, -1] = A[h, -1] / A[h, -1].sum().clamp_min(1e-9)
        return A
    return edit


@torch.no_grad()
def ablate_and_measure(model, W, ids, tok_g, tok_c, S, cands):
    """Live-model margin after jointly ablating `cands`; one forward."""
    with attention_edits(model, W, prob_edit=_edge_edit(cands, S)):
        m, _ = margin(model, ids, tok_g, tok_c)
    return m


@torch.no_grad()
def ground_truth(model, W, ids, tok_g, tok_c, S, candidates, m0,
                 progress=None):
    """Measured effect (m0 - m_ablated) per candidate; one forward each."""
    eff = {}
    for lh in candidates:
        eff[lh] = m0 - ablate_and_measure(model, W, ids, tok_g, tok_c, S,
                                          [lh])
        if progress is not None:
            progress.update(1)
    return eff


def candidate_subset(scores: Dict[str, torch.Tensor], top=20, n_random=40,
                     seed=0) -> List[tuple]:
    """Union of each ranker's top candidates plus a random sample -- the
    measured set for large models (Spearman/precision computed on it)."""
    L, H = scores["affine"].shape
    picked = set()
    for name, s in scores.items():
        if name == "random":
            continue
        idx = torch.topk(s.flatten(), min(top, L * H)).indices
        picked |= {(int(i) // H, int(i) % H) for i in idx}
    g = torch.Generator().manual_seed(seed)
    for i in torch.randperm(L * H, generator=g)[:n_random].tolist():
        picked.add((i // H, i % H))
    return sorted(picked)


# ---------------------- metrics ---------------------------------------------

def magnitude_metrics(scores: Dict[str, torch.Tensor], effects: dict,
                      ks=(4, 8, 16)) -> dict:
    """Magnitude-aware ranking metrics on the measured candidate set.

    Binary precision@k treats a barely-positive component like a decisive
    one; these do not.  Per ranker and k:
      ndcg@k       nDCG with gains = positive measured effects
      orecall@k    recall of the oracle top-k
      capture@k    sum of measured effects of the ranker's top-k, as a
                   fraction of the oracle top-k's sum ('oracle fraction')
    Plus signed evaluation: Spearman restricted to positive (promoting) and
    negative (suppressing) components separately, and sign agreement.
    """
    import math
    from scipy.stats import spearmanr
    cands = list(effects)
    y = torch.tensor([effects[c] for c in cands])
    order_y = torch.argsort(y, descending=True)
    out = {}
    for name, s in scores.items():
        x = torch.tensor([float(s[l, h]) for l, h in cands])
        order_x = torch.argsort(x, descending=True)
        gains = y.clamp_min(0.0)
        for k in ks:
            k = min(k, len(cands))
            dcg = sum(float(gains[order_x[i]]) / math.log2(i + 2)
                      for i in range(k))
            idcg = sum(float(gains[order_y[i]]) / math.log2(i + 2)
                       for i in range(k)) or 1e-9
            out[f"{name}_ndcg@{k}"] = dcg / idcg
            top_o = set(order_y[:k].tolist())
            top_x = set(order_x[:k].tolist())
            out[f"{name}_orecall@{k}"] = len(top_o & top_x) / k
            cap = float(y[list(top_x)].sum())
            ocap = float(y[list(top_o)].sum()) or 1e-9
            out[f"{name}_capture@{k}"] = cap / ocap
        pos, neg = y > 0, y < 0
        for tag_, msk in (("pos", pos), ("neg", neg)):
            if (msk.sum() >= 5 and len(set(y[msk].tolist())) > 1
                    and len(set(x[msk].tolist())) > 1):
                out[f"{name}_rho_{tag_}"] = spearmanr(
                    x[msk], y[msk]).statistic
        out[f"{name}_signagree"] = float(((x > 0) == (y > 0)).float().mean())
    return out


def paired_bootstrap(a, b, n_boot=4000, seed=0):
    """Paired bootstrap CI for mean(a - b) + win rate; a, b same length."""
    import random as _r
    rng = _r.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return dict(mean_diff=sum(d) / n,
                ci_lo=means[int(0.025 * n_boot)],
                ci_hi=means[int(0.975 * n_boot)],
                win_rate=sum(x > 0 for x in d) / n)


def ranking_metrics(scores: Dict[str, torch.Tensor], effects: dict,
                    ks=(1, 2, 4, 8, 16)) -> dict:
    """Per-example Spearman + precision@k on the measured candidate set."""
    from scipy.stats import spearmanr
    cands = list(effects)
    y = [effects[c] for c in cands]
    oracle = [c for _, c in
              sorted(zip(y, cands), key=lambda t: -t[0])]
    out = {}
    for name, s in scores.items():
        x = [float(s[l, h]) for l, h in cands]
        # rho undefined when either side is constant (e.g. attention with no
        # source mass, or a corrupted specificity variant scoring all-zero);
        # record NaN explicitly -- downstream medians exclude NaNs.
        rho = (spearmanr(x, y).statistic
               if len(set(y)) > 1 and len(set(x)) > 1 else float("nan"))
        ranked = [c for _, c in
                  sorted(zip(x, cands), key=lambda t: -t[0])]
        for k in ks:
            k = min(k, len(cands))
            out[f"{name}_p@{k}"] = (len(set(ranked[:k]) & set(oracle[:k]))
                                    / k)
        out[f"{name}_rho"] = rho
    return out


@torch.no_grad()
def cumulative_curves(model, W, ids, tok_g, tok_c, S, scores, m0,
                      effects=None, ks=(1, 2, 4, 8, 16, 32)) -> dict:
    """Joint-ablation margin drop of each ranker's top-k (measured live),
    plus the oracle curve on the measured set if effects are given."""
    L, H = scores["affine"].shape
    out = {}
    rankings = {n: [ (int(i) // H, int(i) % H)
                     for i in torch.argsort(s.flatten(), descending=True)]
                for n, s in scores.items()}
    if effects is not None:
        rankings["oracle"] = [c for _, c in
                              sorted(((v, c) for c, v in effects.items()),
                                     key=lambda t: -t[0])]
    for name, order in rankings.items():
        for k in ks:
            if k > len(order):
                continue
            m = ablate_and_measure(model, W, ids, tok_g, tok_c, S, order[:k])
            out[f"{name}_drop@{k}"] = m0 - m
    return out
