"""Experiment 1: validate the pinned obstruction ob(s).

Per model, over a labeled cohort (your existing verdicts), this driver:
  (a) computes ob(s) and its residual depth profile for every example;
  (b) tests class separation: transport vs {presence, selection, correct}
      (AUC + Mann-Whitney), and presence-vs-transport via the depth centroid;
  (c) checks agreement with the existing transported-support verdicts;
  (d) if per-example ablation effects are supplied (or --run_ablation), compares
      Spearman(ob, effect) against Spearman(attention mass, effect) and
      Spearman(tau, effect);
  (e) writes a per-example CSV so the cross-model table is a concat away.

Cohort schema (jsonl, one record per example):
  {"prompt": str, "target_first_token": int, "competitor_token": int,
   "source_span": [start, end]          # inclusive token indices, contiguous
   "verdict": "presence"|"transport"|"selection"|"correct",
   "tau": float | null,                 # your delivered-transport scalar
   "ablation_effect": float | null}     # corrected logit drop, if precomputed

Usage:
  python run_obstruction_validation.py --model meta-llama/Llama-3.2-3B \
      --cohort cohorts/llama32_3b_parametric.jsonl --out results/llama32_3b
"""
import argparse, json, os
import torch
from scipy.stats import mannwhitneyu, spearmanr

from frozen_cache import (Weights, build_cache, certify_frozen,
                          tdla_edge_scores, target_delivery)
from obstruction import solve_obstruction


def auc(pos, neg):
    """Rank-based AUC for 'pos scores higher than neg'."""
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return max(u, len(pos) * len(neg) - u) / (len(pos) * len(neg))


@torch.no_grad()
def ablation_effect(model, tok_ids, W, C, tok_g, S, k=8, n_random=8):
    """Corrected target-logit drop from ablating the top-k tDLA edges at the
    final position, inside the FROZEN account (fast screen; your Sec 6.9 code
    on the live model is the ground truth -- pass precomputed values if so)."""
    from frozen_cache import frozen_forward, frozen_layer
    scores = tdla_edge_scores(W, C, tok_g, S)               # [L, H, |S|]
    flat = scores.flatten()
    top = torch.topk(flat, k).indices
    Lh, Hh, Sn = scores.shape

    def run_with_zeroed(edges):
        x = C.resid[0]
        for l in range(W.L):
            el = [(h, list(S)[j]) for (ll, h, j) in edges if ll == l]
            def edit(A, el=el):
                if not el:
                    return A
                A = A.clone()
                for h, jj in el:
                    A[h, -1, jj] = 0.0
                A[:, -1, :] = A[:, -1, :] / A[:, -1, :].sum(-1, keepdim=True)
                return A
            x = frozen_layer(x, W, l, C, attn_edit=edit)
        return float(target_delivery(x.unsqueeze(0)[0], W, C, tok_g))

    def unflatten(idx):
        l = idx // (Hh * Sn); r = idx % (Hh * Sn)
        return (int(l), int(r // Sn), int(r % Sn))

    base = float(target_delivery(C.resid[-1], W, C, tok_g))
    drop = base - run_with_zeroed([unflatten(int(i)) for i in top])
    rnd = []
    for _ in range(n_random):
        ridx = torch.randint(0, flat.numel(), (k,))
        rnd.append(base - run_with_zeroed([unflatten(int(i)) for i in ridx]))
    return drop - sum(rnd) / len(rnd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pin_frac", type=float, default=0.5)
    ap.add_argument("--cplus_q", type=float, default=0.10,
                    help="success quantile for c_plus (paper's Q+_{0.10})")
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--run_ablation", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation="eager").to(args.device).eval()
    W = Weights(model)

    records = [json.loads(l) for l in open(args.cohort)]

    # ---- pass 1: caches + calibrate c_plus on the success cohort ----------
    caches, delivered = {}, []
    for i, r in enumerate(records):
        ids = tok(r["prompt"], return_tensors="pt").input_ids[0].to(args.device)
        C = build_cache(model, W, ids)
        certify_frozen(W, C)
        caches[i] = (ids, C)
        if r["verdict"] == "correct":
            delivered.append(float(target_delivery(
                C.resid[-1], W, C, r["target_first_token"])))
    delivered = torch.tensor(delivered)
    c_plus = float(delivered.quantile(args.cplus_q))
    print(f"c_plus (Q_{args.cplus_q:.2f} success delivery) = {c_plus:.3f}  "
          f"[n_success={len(delivered)}]")

    # ---- pass 2: obstruction per example ----------------------------------
    rows = []
    for i, r in enumerate(records):
        ids, C = caches[i]
        S = list(range(r["source_span"][0], r["source_span"][1] + 1))
        res = solve_obstruction(W, C, S, r["target_first_token"], c_plus,
                                pin_L=int(args.pin_frac * W.L), lam=args.lam)
        eff = r.get("ablation_effect")
        if eff is None and args.run_ablation:
            eff = ablation_effect(model, ids, W, C,
                                  r["target_first_token"], S)
        attn_mass = float(sum(C.layers[l].A[:, -1, S].sum()
                              for l in range(W.L)))
        rows.append(dict(idx=i, verdict=r["verdict"], ob=res.ob,
                         ob_norm=res.ob_norm, centroid=res.depth_centroid,
                         gap0=res.delivered_gap0, pi_S=res.pi_S,
                         tau=r.get("tau"), attn_mass=attn_mass,
                         ablation_effect=eff, cg_iters=res.cg_iters))
        print(f"[{i:4d}] {r['verdict']:<9s} ob={res.ob:8.3f} "
              f"norm={res.ob_norm:8.3f} centroid={res.depth_centroid:.2f}")

    import csv
    with open(os.path.join(args.out, "obstruction.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- analysis ---------------------------------------------------------
    by = lambda v, key="ob": [r[key] for r in rows if r["verdict"] == v]
    print("\n=== (a) class separation, ob(s) ===")
    for v in ("presence", "transport", "selection", "correct"):
        xs = torch.tensor(by(v) or [float('nan')])
        print(f"  {v:<9s} n={len(by(v)):3d} median ob={xs.median():8.3f}")
    print(f"  AUC transport vs correct+selection: "
          f"{auc(by('transport'), by('correct') + by('selection')):.3f}")
    print(f"  AUC {{presence,transport}} vs {{selection,correct}}: "
          f"{auc(by('presence') + by('transport'), by('selection') + by('correct')):.3f}")
    print(f"  presence vs transport by DEPTH CENTROID (presence earlier): "
          f"{auc(by('transport', 'centroid'), by('presence', 'centroid')):.3f}")

    print("\n=== (b) agreement with existing transport verdicts ===")
    fails = [r for r in rows if r["verdict"] != "correct"]
    med = torch.tensor([r["ob"] for r in fails]).median()
    pred_transport = [(r["ob"] > med and r["centroid"] > 0.35) for r in fails]
    is_transport = [r["verdict"] == "transport" for r in fails]
    agree = sum(p == t for p, t in zip(pred_transport, is_transport)) / len(fails)
    print(f"  simple ob-threshold agreement with verdicts: {agree:.2f}")

    have = [r for r in rows if r["ablation_effect"] is not None]
    if have:
        print("\n=== (c) predicting ablation effect sizes ===")
        eff = [r["ablation_effect"] for r in have]
        for name, key in (("ob(s)", "ob"), ("attention mass", "attn_mass"),
                          ("tau", "tau")):
            vals = [r[key] for r in have]
            if any(v is None for v in vals):
                continue
            rho, p = spearmanr(vals, eff)
            print(f"  Spearman({name:<15s}, effect) = {rho:+.3f}  (p={p:.1e})")


if __name__ == "__main__":
    main()
