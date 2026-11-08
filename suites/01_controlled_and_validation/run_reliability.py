"""Experiment R4: reliability across equivalent prompts.

For each fact, three NUISANCE presentations (template rewording; trailing
filler; distractor order) and one ROUTE-CHANGE presentation (the answer
supplied in context, which the paper shows switches the model to the
context route).  Measured per pair of presentations:

  map_cos    cosine of the [L,H] transported-support maps
  topk_jac   Jaccard overlap of top-8 candidates
  d_ob       |ob difference| (success-scale units)

Predictions: nuisance pairs give high map_cos / topk_jac (stable account);
route-change pairs give visibly lower similarity (sensitive when the model
genuinely changes strategy).  The instrument should be stable when it ought
to be and sensitive when the computation changes.

Usage:
  python run_reliability.py --model <hf> --cohort <jsonl> --out results/<tag>
"""
import argparse, json, os, csv, itertools
import torch

from frozen_cache import Weights, build_cache, certify_frozen, tdla_edge_scores
from obstruction import solve_obstruction
from build_cohort import find_span

NUISANCE = ["The capital of {s} is",
            "Question: what is the capital of {s}? Answer:",
            "As everyone knows, the capital of {s} is"]
ROUTE = "{a} is a capital city. The capital of {s} is"     # context route


def tmap(W, C, g, c, S):
    return tdla_edge_scores(W, C, g, S, tok_c=c).sum(-1).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_facts", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation="eager").to(args.device).eval()
    W = Weights(model)

    records = [json.loads(l) for l in open(args.cohort)
               if json.loads(l).get("subject") and
               json.loads(l)["competitor_token"] != -1][:args.n_facts]

    rows = []
    for r in tqdm(records, desc="facts"):
        subj = r["subject"]
        ans = r.get("target_text", "").strip() or None
        variants = {}
        prompts = {f"nuis{i}": t.format(s=subj)
                   for i, t in enumerate(NUISANCE)}
        if ans:
            prompts["route"] = ROUTE.format(a=ans, s=subj)
        for name, p in prompts.items():
            try:
                ids = tok(p, return_tensors="pt").input_ids[0].to(args.device)
                C = build_cache(model, W, ids)
                certify_frozen(W, C)
                span = find_span(p, subj, tok)
                S = list(range(span[0], span[1] + 1))
                g, c = r["target_first_token"], r["competitor_token"]
                res = solve_obstruction(W, C, S, g,
                                        c_plus=0.0)   # ob vs fixed ref level
                variants[name] = (tmap(W, C, g, c, S).cpu(), res.ob)
            except (ValueError, AssertionError, IndexError):
                continue
        for a, b in itertools.combinations(variants, 2):
            (ma, oa), (mb, ob_) = variants[a], variants[b]
            n = min(len(ma), len(mb))
            ma, mb = ma[:n], mb[:n]
            cos = float((ma @ mb) / (ma.norm() * mb.norm() + 1e-9))
            ta = set(torch.topk(ma, min(8, n)).indices.tolist())
            tb = set(torch.topk(mb, min(8, n)).indices.tolist())
            rows.append(dict(subject=subj, pair=f"{a}-{b}",
                             kind=("route" if "route" in (a, b)
                                   else "nuisance"),
                             map_cos=cos,
                             topk_jac=len(ta & tb) / len(ta | tb),
                             d_ob=abs(oa - ob_)))

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/reliability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    import statistics as st
    print("\n=== reliability: nuisance vs route-change ===")
    for kind in ("nuisance", "route"):
        sub = [r for r in rows if r["kind"] == kind]
        if sub:
            print(f"  {kind:<9s} n={len(sub):4d}  "
                  f"map_cos {st.median(r['map_cos'] for r in sub):.3f}  "
                  f"top8_jac {st.median(r['topk_jac'] for r in sub):.2f}")
    print("prediction: nuisance similarity >> route-change similarity.")


if __name__ == "__main__":
    main()
