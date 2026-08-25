"""Obstruction stress sweep: does ob(s) respond to delivery difficulty and
stay flat under surface paraphrase?

The v1 reliability d_ob was ~0 because the parametric cohort sits where the
statistic cannot discriminate.  This sweep constructs key-value retrieval
prompts with controlled difficulty:

  distractors  4 / 8 / 16 / 32 pairs
  target slot  early / middle / late
  ambiguity    unique key vs a second matching key with a different value
  paraphrase   two surface forms of the registry header (control)

Per condition: accuracy, delivery, ob(s) distribution at scientific
precision.  Predictions: ob's nonzero rate and magnitude rise with
distractor count and ambiguity; slot has a position effect; paraphrase
changes nothing.  If ob still cannot be made to vary, it is excluded from
reliability claims (a constant statistic is not evidence of robustness).

Usage: python run_ob_stress.py --model <hf> --out results/<tag> --device cuda
"""
import argparse, os, csv, random
import torch

from frozen_cache import Weights, build_cache, certify_frozen, target_delivery
from obstruction import solve_obstruction
from build_cohort import VALUES, KEYS, find_span

HEADERS = ["Registry:", "Lookup table of assignments:"]


def make_prompt(rng, n_pairs, slot, ambiguous, header):
    keys = rng.sample(KEYS * 3, n_pairs)          # allow reuse at n>24
    vals = [rng.choice(VALUES) for _ in keys]
    qi = {"early": 0, "middle": n_pairs // 2, "late": n_pairs - 1}[slot]
    if ambiguous:
        j = (qi + n_pairs // 2) % n_pairs
        keys[j] = keys[qi]                        # same key, different value
    lines = [f"{k}: {v}" for k, v in zip(keys, vals)]
    prompt = header + "\n" + "\n".join(lines) + \
        f"\nThe value of {keys[qi]} is"
    return prompt, vals[qi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_cell", type=int, default=12)
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
    rng = random.Random(0)

    # calibrate c_plus on easy correct prompts
    cal = []
    for _ in range(20):
        p, ans = make_prompt(rng, 4, "late", False, HEADERS[0])
        ids = tok(p, return_tensors="pt").input_ids[0].to(args.device)
        C = build_cache(model, W, ids)
        g = tok(" " + ans, add_special_tokens=False).input_ids[0]
        if int(C.logits[-1].argmax()) == g:
            cal.append(float(target_delivery(C.resid[-1], W, C, g)))
    c_plus = float(torch.tensor(cal).quantile(0.10))
    print(f"c_plus (easy-correct Q0.10) = {c_plus:.3f}  [n={len(cal)}]")

    cells = [(n, slot, amb, h) for n in (4, 8, 16, 32)
             for slot in ("early", "middle", "late")
             for amb in (False, True) for h in range(2)]
    rows = []
    for n, slot, amb, h in tqdm(cells, desc="cells"):
        for _ in range(args.n_per_cell):
            p, ans = make_prompt(rng, n, slot, amb, HEADERS[h])
            try:
                ids = tok(p, return_tensors="pt").input_ids[0].to(args.device)
                if len(ids) > 400:
                    continue
                C = build_cache(model, W, ids)
                certify_frozen(W, C)
                g = tok(" " + ans, add_special_tokens=False).input_ids[0]
                span = find_span(p, ans, tok)     # value span = source
                S = list(range(span[0], span[1] + 1))
                res = solve_obstruction(W, C, S, g, c_plus, cg_iters=80)
                rows.append(dict(
                    n_pairs=n, slot=slot, ambiguous=amb, header=h,
                    T=len(ids), correct=int(C.logits[-1].argmax()) == g,
                    delivery=float(target_delivery(C.resid[-1], W, C, g)),
                    ob=res.ob, gap0=res.delivered_gap0))
            except (ValueError, AssertionError, IndexError):
                continue

    with open(f"{args.out}/ob_stress.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    import statistics as st
    print("\n=== ob(s) stress sweep (scientific precision) ===")
    print(f"{'n':>4s} {'amb':>5s}  {'acc':>5s}  {'ob>0':>5s}  "
          f"{'ob med (nonzero)':>18s}")
    for n in (4, 8, 16, 32):
        for amb in (False, True):
            sub = [r for r in rows if r["n_pairs"] == n
                   and r["ambiguous"] == amb]
            if not sub:
                continue
            nz = [r["ob"] for r in sub if r["ob"] > 1e-9]
            print(f"{n:>4d} {str(amb):>5s}  "
                  f"{st.mean(r['correct'] for r in sub):>5.2f}  "
                  f"{len(nz)/len(sub):>5.2f}  "
                  f"{(st.median(nz) if nz else 0):>18.6e}")
    para = {}
    for h in range(2):
        sub = [r["ob"] for r in rows if r["header"] == h]
        para[h] = st.median(sub) if sub else 0
    print(f"paraphrase control: header-0 med ob {para[0]:.3e} vs "
          f"header-1 {para[1]:.3e} (should match)")
    print("predictions: ob>0 rate and magnitude rise with n_pairs and "
          "ambiguity; paraphrase flat. If flat everywhere, drop d_ob from "
          "reliability claims.")


if __name__ == "__main__":
    main()
