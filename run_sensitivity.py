"""Experiment 3: threshold-sensitivity sweep for the obstruction verdicts.

Recomputes ob(s) over a grid of the two design choices --
  pin_L    in {L/4, L/2, 3L/4}   (how deep the source section is pinned)
  c_plus   in success-quantiles {0.05, 0.10, 0.25}
-- and reports, per configuration: class medians of ob, the transport-AUC,
and agreement with the reference configuration's verdict split.  The paper
claim this supports: the failure decomposition is a property of the
computation, not of the calibration constants.

Usage:
  python run_sensitivity.py --model <hf-name> --cohort <jsonl> --out <dir>
Reuses one cache per example across all 9 configurations.
"""
import argparse, json, os, csv, itertools
import torch

from frozen_cache import Weights, build_cache, certify_frozen, target_delivery
from obstruction import solve_obstruction
from run_obstruction_validation import auc

PIN_FRACS = (0.25, 0.5, 0.75)
QUANTILES = (0.05, 0.10, 0.25)
REF = (0.5, 0.10)


def sweep(model, tok, W, records, device, lam=30.0, pin_fracs=PIN_FRACS,
          quantiles=QUANTILES):
    caches, delivered = {}, []
    for i, r in enumerate(records):
        ids = tok(r["prompt"], return_tensors="pt").input_ids[0].to(device)
        C = build_cache(model, W, ids)
        certify_frozen(W, C)
        caches[i] = C
        if r["verdict"] == "correct":
            delivered.append(float(target_delivery(
                C.resid[-1], W, C, r["target_first_token"])))
    delivered = torch.tensor(delivered)

    results, ob_by_cfg = [], {}
    for pf, q in itertools.product(pin_fracs, quantiles):
        c_plus = float(delivered.quantile(q))
        obs = {}
        for i, r in enumerate(records):
            S = list(range(r["source_span"][0], r["source_span"][1] + 1))
            res = solve_obstruction(W, caches[i], S, r["target_first_token"],
                                    c_plus, pin_L=int(pf * W.L), lam=lam)
            obs[i] = (res.ob, res.depth_centroid)
        ob_by_cfg[(pf, q)] = obs
        g = lambda v: [obs[i][0] for i, r in enumerate(records)
                       if r["verdict"] == v]
        results.append(dict(
            pin_frac=pf, quantile=q, c_plus=c_plus,
            med_presence=float(torch.tensor(g("presence") or [float("nan")]).median()),
            med_transport=float(torch.tensor(g("transport") or [float("nan")]).median()),
            med_selection=float(torch.tensor(g("selection") or [float("nan")]).median()),
            med_correct=float(torch.tensor(g("correct") or [float("nan")]).median()),
            auc_transport=auc(g("transport"), g("correct") + g("selection")),
            auc_pt_vs_sc=auc(g("presence") + g("transport"),
                             g("selection") + g("correct"))))

    # agreement of the ob>0 split with the reference configuration
    ref = ob_by_cfg[REF] if REF in ob_by_cfg else ob_by_cfg[list(ob_by_cfg)[0]]
    for row in results:
        cfg = (row["pin_frac"], row["quantile"])
        obs = ob_by_cfg[cfg]
        agree = [float((obs[i][0] > 1e-6) == (ref[i][0] > 1e-6))
                 for i in obs]
        row["agree_with_ref"] = sum(agree) / len(agree)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
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

    results = sweep(model, tok, W, records, args.device)
    with open(os.path.join(args.out, "sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader(); w.writerows(results)

    print(f"{'pin':>5s} {'q':>5s} {'AUC(tr)':>8s} {'AUC(pt)':>8s} {'agree':>6s}")
    for r in results:
        print(f"{r['pin_frac']:5.2f} {r['quantile']:5.2f} "
              f"{r['auc_transport']:8.3f} {r['auc_pt_vs_sc']:8.3f} "
              f"{r['agree_with_ref']:6.2f}")
    lo = min(r["agree_with_ref"] for r in results)
    print(f"\nminimum agreement with reference config: {lo:.2f}"
          f"  ({'STABLE' if lo >= 0.9 else 'REPORT AS SENSITIVITY'})")


if __name__ == "__main__":
    main()
