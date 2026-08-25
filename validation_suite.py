"""Library for the validation suite of paper Sec. 4.5 (Table 2, Figure 3).

Arch-generic API over the Llama/Qwen path (frozen_cache) and the
GPT-NeoX/Pythia path (frozen_neox):

  load_arch(name, device) -> (model, tok, api) with uniform callables:
      api.build(ids) -> cache          api.certify(C)
      api.delivery(C, tok) -> float    api.final_state(C) -> x_hat [d]
      api.tdla(C, g, S) -> [L,H,|S|]   api.lens(C, l, i, tokens) -> scores
      api.WU, api.L

  Ordered labels (paper Sec. 2.4): decoy-controlled source readability over
  the span (2 consecutive layers), transport via tau_S and its share against
  the within-model Q0.10 of successes, selection as the remaining negative
  margin.

  Readout interventions on the NORMALIZED final state x_hat (Sec. 4.5):
      answer_up   raise <x_hat, u_g/|u_g|> to the successful median
      alt_down    lower <x_hat, u_c/|u_c|> to the successful median
      combined    both edits
      random      norm-matched random-direction control
  Recovery counts only when the gold first token becomes top-ranked in the
  logits recomputed linearly from the edited state.
"""
from dataclasses import dataclass
from typing import Callable, List, Sequence
import torch


# ----------------------------- arch dispatch ---------------------------------

@dataclass
class Api:
    build: Callable
    certify: Callable
    delivery: Callable
    final_state: Callable
    tdla: Callable
    lens: Callable
    WU: torch.Tensor
    L: int


def load_arch(name, device, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation="eager").to(device).eval()
    return model, tok, make_api(model)


def make_api(model):
    if model.config.model_type == "gpt_neox":
        from frozen_neox import (NeoXWeights, build_cache_neox,
                                 certify_frozen_neox, target_delivery_neox,
                                 tdla_neox, final_state, ln_stats, ln_apply)
        W = NeoXWeights(model)

        def lens(C, l, i, tokens):
            x = C.resid[l][i:i + 1]
            xh = ln_apply(x, ln_stats(x, W.eps), W.lnfw, W.lnfb)[0]
            return xh @ W.WU[list(tokens)].T

        return W, Api(build=lambda ids: build_cache_neox(model, W, ids),
                      certify=lambda C: certify_frozen_neox(W, C),
                      delivery=lambda C, t: target_delivery_neox(W, C, t),
                      final_state=lambda C: final_state(W, C),
                      tdla=lambda C, g, S, c=None: tdla_neox(W, C, g, S,
                                                             tok_c=c),
                      lens=lens, WU=W.WU, L=W.L)
    else:
        from frozen_cache import (Weights, build_cache, certify_frozen,
                                  target_delivery, tdla_edge_scores,
                                  apply_norm, norm_inv)
        W = Weights(model)

        def final_state_l(C):
            x = C.resid[-1][-1:]
            return apply_norm(x, C.inv_f[-1:], W.ln_f, W.center)[0]

        def lens(C, l, i, tokens):
            x = C.resid[l][i:i + 1]
            xh = apply_norm(x, norm_inv(x, W.eps, W.center), W.ln_f,
                            W.center)[0]
            return xh @ W.WU[list(tokens)].T

        return W, Api(build=lambda ids: build_cache(model, W, ids),
                      certify=lambda C: certify_frozen(W, C),
                      delivery=lambda C, t: float(target_delivery(
                          C.resid[-1], W, C, t)),
                      final_state=final_state_l,
                      tdla=lambda C, g, S, c=None: tdla_edge_scores(
                          W, C, g, S, tok_c=c),
                      lens=lens, WU=W.WU, L=W.L)


# ----------------------------- ordered labels --------------------------------

def readable_over_span(api: Api, C, g: int, decoys: Sequence[int],
                       S: Sequence[int], consecutive=2) -> bool:
    """Target outranks every same-type decoy for >= `consecutive` layers at
    some position in the span (paper Eq. 10)."""
    toks = [g] + [d for d in decoys if d != g]
    if len(toks) < 2:
        return True
    for i in S:
        run = 0
        for l in range(1, api.L + 1):
            s = api.lens(C, l, i, toks)
            run = run + 1 if bool(s[0] > s[1:].max()) else 0
            if run >= consecutive:
                return True
    return False


@dataclass
class Thresholds:
    tau_q10: float
    share_q10: float


def calibrate(api: Api, caches: dict, successes: List[dict]) -> Thresholds:
    taus, shares = [], []
    for r in successes:
        C = caches[r["idx"]]
        S = list(range(r["source_span"][0], r["source_span"][1] + 1))
        tau = float(api.tdla(C, r["target_first_token"], S).sum())
        d = abs(api.delivery(C, r["target_first_token"])) + 1e-9
        taus.append(tau)
        shares.append(tau / d)
    q = lambda v: float(torch.tensor(v).quantile(0.10))
    return Thresholds(tau_q10=q(taus), share_q10=q(shares))


def label_failure(api: Api, C, r: dict, decoys, th: Thresholds,
                  token_identity=False) -> str:
    g, c = r["target_first_token"], r["competitor_token"]
    S = list(range(r["source_span"][0], r["source_span"][1] + 1))
    if not token_identity and not readable_over_span(api, C, g, decoys, S):
        return "source"
    tau = float(api.tdla(C, g, S).sum())
    share = tau / (abs(api.delivery(C, g)) + 1e-9)
    if tau < th.tau_q10 or share < th.share_q10:
        return "transport"
    return "selection"


# ----------------------------- readout interventions -------------------------

@dataclass
class Medians:
    proj_g: float          # successful median projection onto own gold dir
    proj_c: float          # successful median projection onto runner-up dir


def success_medians(api: Api, caches, successes) -> Medians:
    pg, pc = [], []
    for r in successes:
        C = caches[r["idx"]]
        xh = api.final_state(C)
        ug = api.WU[r["target_first_token"]]
        pg.append(float((xh @ ug) / ug.norm()))
        logits = xh @ api.WU.T
        runner = int(logits.argsort()[-2])
        uc = api.WU[runner]
        pc.append(float((xh @ uc) / uc.norm()))
    med = lambda v: float(torch.tensor(v).median())
    return Medians(proj_g=med(pg), proj_c=med(pc))


def _project_to(xh, u, target_proj):
    uhat = u / u.norm()
    return xh + (target_proj - float(xh @ uhat)) * uhat


def readout_interventions(api: Api, C, g: int, c: int, med: Medians,
                          seed=0) -> dict:
    """Recovery (gold first token top-ranked) under each edit of x_hat."""
    xh = api.final_state(C)
    ug, uc = api.WU[g], api.WU[c]
    up = _project_to(xh, ug, med.proj_g)
    down = _project_to(xh, uc, med.proj_c)
    both = _project_to(up, uc, med.proj_c)
    disp = (up - xh).norm()
    gdir = torch.randn(xh.shape, generator=torch.Generator().manual_seed(seed)
                       ).to(xh.device)
    rand = xh + disp * gdir / gdir.norm()
    top = lambda v: int((v @ api.WU.T).argmax()) == g
    return dict(answer_up=top(up), alt_down=top(down), combined=top(both),
                random=top(rand), base=top(xh))
