"""Experiment 2: the repair matrix.

Applies every repair family to every verdict class and reports the matrix of
flip rates and mean margin changes.  The taxonomy's causal claim is that the
DIAGONAL dominates:

                      source-repair  transport-repair  selection-repair
  source failures        HIGH              low               low
  transport failures       (any)             HIGH              low
  selection failures       (any)             low               HIGH

plus random controls (random heads / random edges / donor-free band patch)
that should sit near zero everywhere.

Cohort schema extends experiment 1's jsonl with donor info for source rows:
  "donor_prompt": str | null      # matched success: same fact, different
                                  # template or with supporting context
  "donor_source_span": [a, b]     # donor's source-span token indices

Usage:
  python run_repair_matrix.py --model meta-llama/Llama-3.2-3B \
      --cohort cohorts/llama32_3b_parametric.jsonl --out results/llama32_3b
"""
import argparse, json, os, csv
import torch

from frozen_cache import Weights, build_cache, certify_frozen
from repairs import (repair_source, repair_transport,
                     repair_transport_force, repair_selection,
                     repair_random_heads, certify_wrappers, margin)

REPAIRS = ("source_patch", "transport_edges", "transport_force",
           "selection_demoters", "random_heads")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band_frac", nargs=2, type=float, default=[0.2, 0.6],
                    help="layer band for the source source patch")
    ap.add_argument("--k_edges", type=int, default=8)
    ap.add_argument("--k_heads", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation="eager").to(args.device).eval()
    W = Weights(model)
    band = list(range(int(args.band_frac[0] * W.L),
                      int(args.band_frac[1] * W.L)))

    records = [json.loads(l) for l in open(args.cohort)
               if json.loads(l)["verdict"] != "correct"]

    rows = []
    certified = False
    for i, r in enumerate(records):
        ids = tok(r["prompt"], return_tensors="pt").input_ids[0].to(args.device)
        C = build_cache(model, W, ids)
        certify_frozen(W, C)
        if not certified:
            certify_wrappers(model, W, ids)
            certified = True
        g, c = r["target_first_token"], r["competitor_token"]
        S = list(range(r["source_span"][0], r["source_span"][1] + 1))
        m0, _ = margin(model, ids, g, c)
        out = dict(idx=i, verdict=r["verdict"], margin_base=m0)

        # --- source repair: donor source-state patch ---------------------
        if r.get("donor_prompt"):
            d_ids = tok(r["donor_prompt"],
                        return_tensors="pt").input_ids[0].to(args.device)
            D = build_cache(model, W, d_ids)
            dS = list(range(r["donor_source_span"][0],
                            r["donor_source_span"][1] + 1))
            assert len(dS) == len(S), \
                "align donor/receiver source spans (pad or re-tokenize)"
            donor = {l: D.resid[l][dS] for l in band}
            m, flip = repair_source(model, ids, g, c, S, donor, band)
            out["source_patch"] = m - m0
            out["source_patch_flip"] = flip
            # control: patch a random non-source band of same width
            rnd_pos = [(p + 3) % ids.shape[0] for p in S]
            m_r, _ = repair_source(model, ids, g, c, rnd_pos, donor, band)
            out["source_patch_ctrl"] = m_r - m0

        # --- transport repairs: weak (boost) and strong (force) ------------
        m, flip = repair_transport(model, W, C, ids, g, c, S, k=args.k_edges)
        out["transport_edges"] = m - m0
        out["transport_edges_flip"] = flip
        m, flip = repair_transport_force(model, W, C, ids, g, c, S,
                                         k=args.k_edges, alpha=1.0)
        out["transport_force"] = m - m0
        out["transport_force_flip"] = flip

        # --- selection repair: ablate demoter heads ------------------------
        m, flip = repair_selection(model, W, C, ids, g, c, k=args.k_heads)
        out["selection_demoters"] = m - m0
        out["selection_demoters_flip"] = flip

        # --- control -------------------------------------------------------
        m, flip = repair_random_heads(model, W, ids, g, c, k=args.k_heads,
                                      seed=i)
        out["random_heads"] = m - m0
        out["random_heads_flip"] = flip

        rows.append(out)
        print(f"[{i:4d}] {r['verdict']:<9s} " + "  ".join(
            f"{k}={out.get(k, float('nan')):+.2f}" for k in REPAIRS
            if k in out))

    with open(os.path.join(args.out, "repair_matrix.csv"), "w",
              newline="") as f:
        keys = sorted({k for row in rows for k in row})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    # ---- the matrix -------------------------------------------------------
    print("\n=== repair matrix: mean margin change (flip rate) ===")
    verdicts = ("source", "transport", "selection")
    header = f"{'':<12s}" + "".join(f"{r:>22s}" for r in REPAIRS)
    print(header)
    for v in verdicts:
        sub = [r for r in rows if r["verdict"] == v]
        line = f"{v:<12s}"
        for rep in REPAIRS:
            vals = [r[rep] for r in sub if rep in r]
            flips = [r.get(rep + "_flip", False) for r in sub if rep in r]
            if vals:
                line += f"  {sum(vals)/len(vals):+8.2f} ({sum(flips)/len(flips):4.2f})"
            else:
                line += f"  {'--':>18s}"
        print(line)
    print("\nDiagonal should dominate; random_heads column should be ~0.")


if __name__ == "__main__":
    main()
