"""Experiment R5: practicality and certificate coverage.

Per model, over the cohort: certificate pass RATES and error DISTRIBUTIONS
(not just representative values), runtime of each instrument relative to one
forward pass, peak memory, sequence-length scaling, and fp32-vs-bf16
certificate reliability.  The engineering table for the tool paper.

Usage:
  python run_profile.py --model <hf> --cohort <jsonl> --out results/<tag>
"""
import argparse, json, os, csv, time
import torch

from frozen_cache import (Weights, build_cache, certify_frozen, frozen_forward,
                          apply_norm, tdla_edge_scores)
from obstruction import solve_obstruction


def timed(fn, *a, **kw):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_examples", type=int, default=40)
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

    records = [json.loads(l) for l in open(args.cohort)][:args.n_examples]
    rows = []
    for r in tqdm(records, desc="profile"):
        ids = tok(r["prompt"], return_tensors="pt").input_ids[0].to(args.device)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _, t_fwd = timed(lambda: model(ids.view(1, -1), use_cache=False))
        C, t_cache = timed(build_cache, model, W, ids)
        try:
            c2 = certify_frozen(W, C)
            c2_pass = True
        except AssertionError:
            c2, c2_pass = float("nan"), False
        S = list(range(r["source_span"][0], r["source_span"][1] + 1))
        res, t_ob = timed(solve_obstruction, W, C, S,
                          r["target_first_token"], c_plus=1e9, cg_iters=50)
        _, t_tdla = timed(tdla_edge_scores, W, C, r["target_first_token"], S)
        peak = (torch.cuda.max_memory_allocated() / 2**30
                if args.device == "cuda" else float("nan"))
        rows.append(dict(idx=r.get("idx"), T=len(ids), c2_err=c2,
                         c2_pass=c2_pass, c3_err=res.cert_affine,
                         t_forward=t_fwd, t_cache=t_cache, t_ob=t_ob,
                         t_tdla=t_tdla, x_cache=t_cache / t_fwd,
                         x_ob=t_ob / t_fwd, x_tdla=t_tdla / t_fwd,
                         peak_gb=peak))

    # ---- bf16 reliability (cache in bf16, certify against fp32 model) ------
    bf_errs = []
    try:
        model_bf = model.to(torch.bfloat16)
        for r in records[:10]:
            ids = tok(r["prompt"],
                      return_tensors="pt").input_ids[0].to(args.device)
            model_fp = model_bf.float()
            try:
                C = build_cache(model_fp, Weights(model_fp), ids,
                                cert_tol=1e9)
                ref = model_fp(ids.view(1, -1)).logits[0].float()
                bf_errs.append(float((C.logits - ref).norm() / ref.norm()))
            finally:
                model_bf = model_fp.to(torch.bfloat16)
        model = model_bf.float()
    except RuntimeError:
        pass

    with open(f"{args.out}/profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    import statistics as st
    n = len(rows)
    print(f"\n=== practicality: {args.model} ===")
    print(f"  certificate pass rate (C2): "
          f"{sum(r['c2_pass'] for r in rows)}/{n}")
    print(f"  C2 rel err   median {st.median(r['c2_err'] for r in rows):.2e} "
          f"max {max(r['c2_err'] for r in rows):.2e}")
    print(f"  C3 affinity  median {st.median(r['c3_err'] for r in rows):.2e}")
    print(f"  runtime (x one forward): cache "
          f"{st.median(r['x_cache'] for r in rows):.1f}  "
          f"R(S) {st.median(r['x_ob'] for r in rows):.0f}  "
          f"tDLA {st.median(r['x_tdla'] for r in rows):.1f}")
    if rows[0]['peak_gb'] == rows[0]['peak_gb']:
        print(f"  peak memory: {max(r['peak_gb'] for r in rows):.1f} GB")
    if bf_errs:
        print(f"  fp32-cache-vs-fp32-model err after bf16 round trip: "
              f"median {st.median(bf_errs):.2e}")


if __name__ == "__main__":
    main()
