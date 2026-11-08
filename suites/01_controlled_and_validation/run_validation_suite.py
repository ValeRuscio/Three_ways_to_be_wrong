"""Paper Sec. 4.5: diagnostic labels + calibrated readout interventions
(Table 2, Figure 3), over eight models including OLMo 7B and Pythia 2.8B.

Per model: 60 failed + 60 successful generations from six PopQA relations;
ordered labels via the decoy-controlled lens + calibrated transport; then
answer-up / alternative-down / combined / norm-matched-random edits of the
normalized final state, with recovery = gold first token becomes top-ranked.

Statistics printed at the end: pooled shares (Table 2), recovery by label
with Wilson intervals, Cochran-Armitage trend for answer-up across the
stage order, Holm-corrected source-selection comparison, and the logistic
model with model fixed effects (odds ratio per stage).

Usage:
  python run_validation_suite.py --model EleutherAI/pythia-2.8b \
      --cohort cohorts/pythia-28b_parametric.jsonl --out results/pythia-28b
  python run_validation_suite.py --aggregate     # after all models
"""
import argparse, json, os, csv, glob
import torch

from validation_suite import (load_arch, calibrate, label_failure,
                              readable_over_span, success_medians,
                              readout_interventions)
from stats_paper import (wilson, cochran_armitage, holm, two_prop_p,
                         logistic_fixed_effects)

STAGE_ORDER = {"source": 0, "transport": 1, "selection": 2}


def run_model(model_name, cohort_path, out_dir, device):
    from tqdm import tqdm
    os.makedirs(out_dir, exist_ok=True)
    model, tok, (W, api) = load_arch(model_name, device)

    records = [json.loads(l) for l in open(cohort_path)]
    fails = [r for r in records if not r["correct"]
             and r["competitor_token"] != -1][:60]
    succs = [r for r in records if r["correct"]][:60]
    use = fails + succs
    for i, r in enumerate(use):
        r["idx"] = i

    caches = {}
    for r in tqdm(use, desc=f"{os.path.basename(out_dir)} caches"):
        ids = tok(r["prompt"], return_tensors="pt").input_ids[0].to(device)
        C = api.build(ids)
        api.certify(C)
        caches[r["idx"]] = C

    th = calibrate(api, caches, succs)
    med = success_medians(api, caches, succs)

    # decoys: same-relation targets when available, else all cohort targets
    def decoys_for(r):
        rel = r.get("relation")
        pool = [x["target_first_token"] for x in records
                if (rel is None or x.get("relation") == rel)
                and x["target_first_token"] != r["target_first_token"]]
        return list(dict.fromkeys(pool))[:20]

    rows = []
    for r in tqdm(fails, desc="labels+interventions"):
        C = caches[r["idx"]]
        lab = label_failure(api, C, r, decoys_for(r), th,
                            token_identity=r.get("task") == "extraction")
        rec = readout_interventions(api, C, r["target_first_token"],
                                    r["competitor_token"], med, seed=r["idx"])
        rows.append(dict(idx=r["idx"], label=lab,
                         stage=STAGE_ORDER[lab],
                         s_pop=r.get("s_pop"),
                         target_tok=r["target_first_token"],
                         competitor_tok=r["competitor_token"], **rec))
    with open(os.path.join(out_dir, "validation_suite.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    shares = Counter(r["label"] for r in rows)
    n = len(rows)
    print(f"shares: " + "  ".join(f"{k}={v/n:.2f}" for k, v in
                                  sorted(shares.items())))
    return rows


def aggregate():
    import pandas as pd
    frames = []
    for p in sorted(glob.glob("results/*/validation_suite.csv")):
        frames.append(pd.read_csv(p).assign(model=p.split(os.sep)[-2]))
    df = pd.concat(frames)
    n = len(df)
    print(f"\n=== validation suite: {df.model.nunique()} models, "
          f"{n} failures ===")
    print("\n[Table 2] label shares per model:")
    print(df.groupby("model").label.value_counts(normalize=True)
            .unstack(fill_value=0).round(2))
    pooled = df.label.value_counts(normalize=True)
    print("pooled:", {k: round(v, 2) for k, v in pooled.items()})

    print("\n[Figure 3] recovery by label (Wilson 95%):")
    labs = ["source", "transport", "selection"]
    for arm in ("answer_up", "combined", "alt_down", "random"):
        line = f"  {arm:<10s}"
        for lab in labs:
            sub = df[df.label == lab]
            k, m = int(sub[arm].sum()), len(sub)
            lo, hi = wilson(k, m)
            line += f"  {lab[:4]} {k/m:.3f} [{lo:.3f},{hi:.3f}]"
        print(line)

    ks = [int(df[df.label == l].answer_up.sum()) for l in labs]
    ns = [len(df[df.label == l]) for l in labs]
    z, p = cochran_armitage(ks, ns)
    print(f"\nCochran-Armitage trend (answer_up): z={z:.2f}, p={p:.4f}")
    pw = {"source-selection": two_prop_p(ks[0], ns[0], ks[2], ns[2]),
          "source-transport": two_prop_p(ks[0], ns[0], ks[1], ns[1]),
          "transport-selection": two_prop_p(ks[1], ns[1], ks[2], ns[2])}
    print("Holm-corrected pairwise:", {k: round(v, 4)
                                       for k, v in holm(pw).items()})
    lf = logistic_fixed_effects(df.answer_up.astype(float).tolist(),
                                df.stage.tolist(), df.model.tolist())
    print(f"logistic w/ model FE: OR per stage {lf['odds_ratio']:.2f} "
          f"(p={lf['p']:.4f})")
    d_sel = (df[df.label == 'selection'].combined.mean()
             - df[df.label == 'selection'].answer_up.mean())
    d_oth = max((df[df.label == l].combined.mean()
                 - df[df.label == l].answer_up.mean())
                for l in ("source", "transport"))
    print(f"alt-down increment: selection +{100*d_sel:.1f} pts vs "
          f"earlier stages at most +{100*d_oth:.1f} pts")
    print(f"random control recovery: {df.random.mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--cohort")
    ap.add_argument("--out")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run_model(args.model, args.cohort, args.out, args.device)


if __name__ == "__main__":
    main()
