"""Experiment R1: blinded causal ranking (+ specificity controls).

Per model x example: freeze six rankings from the cached pass, measure
ground-truth single-candidate effects on the live model, compute
Spearman / precision@k / cumulative-drop / regret curves.

--specificity additionally rescoring the SHEAF ranker under corrupted
configurations (same-type wrong target; shifted source span; random target)
against the same measured ground truth: certificates still pass, Spearman
should collapse -- exactness is not specificity.

Usage:
  python run_ranking.py --model meta-llama/Llama-3.2-1B \
      --cohort cohorts/llama-32-1b_parametric.jsonl --out results/llama-32-1b \
      --n_examples 25 [--full] [--specificity]
"""
import argparse, json, os, csv, random
import torch

from frozen_cache import Weights, build_cache, certify_frozen, tdla_edge_scores
from repairs import certify_wrappers, margin
from ranking import (score_candidates, ground_truth, candidate_subset,
                     ranking_metrics, magnitude_metrics,
                     cumulative_curves, RANKERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_examples", type=int, default=50)
    ap.add_argument("--full", action="store_true",
                    help="measure ALL LxH candidates (small models)")
    ap.add_argument("--specificity", action="store_true")
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
               if json.loads(l)["competitor_token"] != -1]
    rng = random.Random(0)
    fails = [r for r in records if r["verdict"] != "correct"]
    corrs = [r for r in records if r["verdict"] == "correct"]
    n2 = args.n_examples // 2
    sample = (rng.sample(fails, min(n2, len(fails))) +
              rng.sample(corrs, min(args.n_examples - n2, len(corrs))))

    rows, spec_rows = [], []
    for i, r in enumerate(tqdm(sample, desc="examples")):
        ids = tok(r["prompt"], return_tensors="pt").input_ids[0].to(args.device)
        C = build_cache(model, W, ids)
        certify_frozen(W, C)
        if i == 0:
            certify_wrappers(model, W, ids)
        g, c = r["target_first_token"], r["competitor_token"]
        S = list(range(r["source_span"][0], r["source_span"][1] + 1))
        m0, _ = margin(model, ids, g, c)

        scores = score_candidates(model, W, C, ids, g, c, S, seed=i)
        cands = ([(l, h) for l in range(W.L) for h in range(W.H)]
                 if args.full else candidate_subset(scores, seed=i))
        bar = tqdm(total=len(cands), leave=False, desc="ablate")
        effects = ground_truth(model, W, ids, g, c, S, cands, m0,
                               progress=bar)
        bar.close()

        row = dict(idx=i, verdict=r["verdict"], correct=r["verdict"] == "correct",
                   n_cands=len(cands), m0=m0)
        row.update(ranking_metrics(scores, effects))
        row.update(magnitude_metrics(scores, effects))
        row.update(cumulative_curves(model, W, ids, g, c, S, scores, m0,
                                     effects=effects))
        rows.append(row)

        if args.specificity:
            wrong_tgt = rng.choice([x["target_first_token"] for x in records
                                    if x["target_first_token"] != g])
            shift = min(len(ids) - S[-1] - 2, 3) or -min(S[0], 3)
            variants = {
                "true": scores["sheaf"],
                "wrong_target": tdla_edge_scores(W, C, wrong_tgt, S,
                                                 tok_c=c).sum(-1).cpu(),
                "wrong_span": tdla_edge_scores(W, C, g,
                                               [p + shift for p in S],
                                               tok_c=c).sum(-1).cpu(),
                "random_target": tdla_edge_scores(
                    W, C, rng.randrange(W.WU.shape[0]), S,
                    tok_c=c).sum(-1).cpu(),
            }
            sr = {"idx": i}
            sr.update({f"rho_{k}": ranking_metrics({"x": v}, effects)["x_rho"]
                       for k, v in variants.items()})
            spec_rows.append(sr)

    with open(os.path.join(args.out, "ranking.csv"), "w", newline="") as f:
        keys = sorted({k for row in rows for k in row})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    if spec_rows:
        with open(os.path.join(args.out, "specificity.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(spec_rows[0]))
            w.writeheader(); w.writerows(spec_rows)

    # ---- summary -----------------------------------------------------------
    import statistics as st
    print("\n=== blinded ranking: median Spearman(predicted, measured) ===")
    for name in RANKERS:
        vals = [row[f"{name}_rho"] for row in rows
                if row.get(f"{name}_rho") == row.get(f"{name}_rho")]
        print(f"  {name:<10s} {st.median(vals):+.3f}   "
              f"p@8 {st.median([row[f'{name}_p@8'] for row in rows]):.2f}")
    if spec_rows:
        print("=== specificity: sheaf rho under corrupted configs ===")
        for k in ("true", "wrong_target", "wrong_span", "random_target"):
            vals = [s[f"rho_{k}"] for s in spec_rows
                    if s[f"rho_{k}"] == s[f"rho_{k}"]]
            print(f"  {k:<14s} {st.median(vals):+.3f}")


if __name__ == "__main__":
    main()
