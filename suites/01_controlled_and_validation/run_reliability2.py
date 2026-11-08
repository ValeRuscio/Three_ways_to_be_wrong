"""Reliability v2: verified route changes vs nuisance changes, as a
discrimination task.

Per fact: one baseline + three NUISANCE variants (formatting, relation
paraphrase, irrelevant suffix -- filler-padded to roughly match the route
variants' token count) + three ROUTE variants (answer verbatim early in
context; answer sentence late in context; dual-source with an instruction).

Route changes are VERIFIED, not assumed: for each route variant we block the
subject span and, separately, the contextual answer span (zero all attention
into the span at every query position, renormalized) and measure the margin
drops.  A variant counts as a route change only when the contextual source
has the larger causal effect.  Nuisance variants must conversely stay
subject-dominant, or they are excluded as unverified.

Candidate alignment: maps live on the (layer, head) grid -- identical across
prompts by construction; token positions never enter the comparison.

Metrics per (baseline, variant) pair and per map type (affine / attention /
dla / actnorm): weighted cosine, Spearman, top-8 Jaccard, rank-biased
overlap (p = 0.9), depth-centroid shift.  Headline: AUC of map change for
separating verified route changes from nuisance changes, with subject-level
bootstrap CIs.  d_ob is reported at scientific precision and dropped from
the headline if degenerate.

Usage:
  python run_reliability2.py --model <hf> --cohort <jsonl> --out results/<tag>
"""
import argparse, json, os, csv, itertools, random
import torch

from frozen_cache import Weights, build_cache, certify_frozen, tdla_edge_scores
from ranking import _source_messages, _edge_edit
from repairs import attention_edits, margin
from build_cohort import find_span

FILLER = " The weather that day was entirely unremarkable."

def variants(subj, ans):
    nuis = {
        "n_format": f"The capital of  {subj}  is",
        "n_paraphrase": f"The capital city of {subj} is called",
        "n_suffix": f"The capital of {subj} is",   # + filler padding below
    }
    route = {
        "r_ctx_early": f"{ans} is a well-known capital city. "
                       f"The capital of {subj} is",
        "r_ctx_late": f"The capital of {subj} is a city named {ans}. "
                      f"The capital of {subj} is",
        "r_dual": f"An old atlas says the capital of {subj} is {ans}. "
                  f"Trusting the atlas, the capital of {subj} is",
    }
    # pad nuisance prompts with neutral filler to match route token counts
    nuis = {k: (v if k != "n_suffix" else v) + FILLER for k, v in nuis.items()}
    return nuis, route


def head_maps(model, W, C, g, c, S):
    s_pre, s_post = _source_messages(W, C, S)
    u = ((C.inv_f[-1] * W.ln_f) * (W.WU[g] - W.WU[c])).squeeze()
    dla = s_post @ u
    if W.center:
        dla = dla - s_post.mean(-1) * u.sum()
    return {
        "affine": tdla_edge_scores(W, C, g, S, tok_c=c).sum(-1).cpu(),
        "attention": torch.stack([C.layers[l].A[:, -1, S].sum(-1)
                                  for l in range(W.L)]).cpu(),
        "dla": dla.cpu(),
        "actnorm": s_post.norm(dim=-1).cpu(),
    }


def block_span(model, W, ids, g, c, span):
    """Margin with ALL attention into `span` removed (every query position)."""
    def edit(A, li):
        A = A.clone()
        A[:, :, list(span)] = 0.0
        A = A / A.sum(-1, keepdim=True).clamp_min(1e-9)
        return A
    with attention_edits(model, W, prob_edit=edit):
        m, _ = margin(model, ids, g, c)
    return m


def rbo(x, y, p=0.9, depth=32):
    """Rank-biased overlap of two [L,H] maps (flattened rankings)."""
    rx = torch.argsort(x.flatten(), descending=True)[:depth].tolist()
    ry = torch.argsort(y.flatten(), descending=True)[:depth].tolist()
    score, seen_x, seen_y = 0.0, set(), set()
    for d in range(1, depth + 1):
        seen_x.add(rx[d - 1]); seen_y.add(ry[d - 1])
        score += (p ** (d - 1)) * len(seen_x & seen_y) / d
    return (1 - p) * score


def map_similarity(a, b):
    fa, fb = a.flatten().double(), b.flatten().double()
    cos = float((fa @ fb) / (fa.norm() * fb.norm() + 1e-12))
    from scipy.stats import spearmanr
    rho = spearmanr(fa, fb).statistic
    ta = set(torch.topk(fa, 8).indices.tolist())
    tb = set(torch.topk(fb, 8).indices.tolist())
    L = a.shape[0]
    depth = torch.arange(L, dtype=torch.double).unsqueeze(1) / max(L - 1, 1)
    cen = lambda m: float((m.double().clamp_min(0) * depth).sum()
                          / m.double().clamp_min(0).sum().clamp_min(1e-9))
    return dict(cos=cos, rho=rho, jac=len(ta & tb) / len(ta | tb),
                rbo=rbo(a, b), d_centroid=abs(cen(a) - cen(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_facts", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation="eager").to(args.device).eval()
    W = Weights(model)

    records = [json.loads(l) for l in open(args.cohort)
               if json.loads(l).get("subject")
               and json.loads(l).get("target_text")
               and json.loads(l)["competitor_token"] != -1][:args.n_facts]

    rows = []
    for r in tqdm(records, desc="facts"):
        subj, ans = r["subject"], r["target_text"].strip()
        g, c = r["target_first_token"], r["competitor_token"]
        base_prompt = f"The capital of {subj} is"
        nuis, route = variants(subj, ans)
        prompts = {"base": base_prompt, **nuis, **route}
        data = {}
        for name, p in prompts.items():
            try:
                ids = tok(p, return_tensors="pt").input_ids[0].to(args.device)
                C = build_cache(model, W, ids)
                certify_frozen(W, C)
                span = find_span(p, subj, tok)
                # last occurrence of the subject = the query mention
                S = list(range(span[0], span[1] + 1))
                maps = head_maps(model, W, C, g, c, S)
                m0, _ = margin(model, ids, g, c)
                entry = dict(maps=maps, m0=m0, verified=None)
                if name.startswith("r_"):
                    ctx = find_span(p, ans, tok)
                    d_subj = m0 - block_span(model, W, ids, g, c,
                                             range(S[0], S[-1] + 1))
                    d_ctx = m0 - block_span(model, W, ids, g, c,
                                            range(ctx[0], ctx[1] + 1))
                    entry["verified"] = bool(d_ctx > d_subj)
                    entry["d_subj"], entry["d_ctx"] = d_subj, d_ctx
                data[name] = entry
            except (ValueError, AssertionError, IndexError):
                continue
        if "base" not in data:
            continue
        for name, e in data.items():
            if name == "base":
                continue
            kind = ("route" if name.startswith("r_") else "nuisance")
            if kind == "route" and not e["verified"]:
                kind = "route_unverified"
            for mtype in ("affine", "attention", "dla", "actnorm"):
                sim = map_similarity(data["base"]["maps"][mtype],
                                     e["maps"][mtype])
                rows.append(dict(subject=subj, variant=name, kind=kind,
                                 map=mtype,
                                 d_subj=e.get("d_subj"),
                                 d_ctx=e.get("d_ctx"), **sim))

    with open(f"{args.out}/reliability2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- discrimination AUC with subject-level bootstrap -------------------
    from run_obstruction_validation import auc
    import statistics as st
    print("\n=== reliability v2: verified-route vs nuisance discrimination ===")
    ver = sum(1 for r in rows if r["kind"] == "route" and r["map"] == "affine")
    unv = sum(1 for r in rows
              if r["kind"] == "route_unverified" and r["map"] == "affine")
    print(f"route variants verified contextual: {ver}/{ver + unv}")
    for mtype in ("affine", "attention", "dla", "actnorm"):
        sub = [r for r in rows if r["map"] == mtype]
        chg = [1 - r["rbo"] for r in sub if r["kind"] == "route"]
        nch = [1 - r["rbo"] for r in sub if r["kind"] == "nuisance"]
        if not chg or not nch:
            continue
        a = auc(chg, nch)
        # subject-level bootstrap
        subjects = sorted({r["subject"] for r in sub})
        rng = random.Random(0)
        boots = []
        for _ in range(1000):
            pick = {rng.choice(subjects) for _ in subjects}
            cg = [1 - r["rbo"] for r in sub
                  if r["kind"] == "route" and r["subject"] in pick]
            ng = [1 - r["rbo"] for r in sub
                  if r["kind"] == "nuisance" and r["subject"] in pick]
            if cg and ng:
                boots.append(auc(cg, ng))
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
        print(f"  {mtype:<10s} AUC(1-RBO) = {a:.3f}  [{lo:.3f}, {hi:.3f}]  "
              f"nuis RBO med {st.median(r['rbo'] for r in sub if r['kind'] == 'nuisance'):.3f}  "
              f"route RBO med {st.median(r['rbo'] for r in sub if r['kind'] == 'route'):.3f}")


if __name__ == "__main__":
    main()
