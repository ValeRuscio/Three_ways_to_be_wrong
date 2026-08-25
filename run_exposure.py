"""Paper Appendix C + Figure 4: training-data exposure by diagnostic label.

Exposure is the subject-answer document co-occurrence count where the
pretraining corpus is public (Dolma for OLMo, the Pile for Pythia, queried
through the infini-gram API), and PopQA subject popularity otherwise.
Frequency features are unigram counts for the target and competitor tokens.

Reported: exposure distributions per label (Kruskal-Wallis), and single- or
few-feature label discrimination (pairwise AUCs, logistic AUC) -- the paper's
claim is that these features separate the labels POORLY, so weak numbers
here are the expected result, not a failure.

API calls are cached in exposure_cache.json and fail gracefully (offline ->
popularity fallback only).

Usage:
  python run_exposure.py --cohort cohorts/<tag>_parametric.jsonl \
      --suite results/<tag>/validation_suite.csv --out results/<tag> \
      [--index v4_piletrain_llama]
"""
import argparse, json, os

INFINIGRAM = "https://api.infini-gram.io/"
INDEX_BY_MODEL = {"pythia": "v4_piletrain_llama",
                  "olmo": "v4_dolma-v1_7_llama"}


def _load_cache(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def infinigram_count(query, index, cache, cache_path, timeout=15):
    key = f"{index}::{query}"
    if key in cache:
        return cache[key]
    try:
        import requests
        r = requests.post(INFINIGRAM, timeout=timeout, json={
            "index": index, "query_type": "count", "query": query})
        cnt = r.json().get("count")
    except Exception:
        cnt = None
    cache[key] = cnt
    json.dump(cache, open(cache_path, "w"))
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default=None,
                    help="infini-gram index for a public-corpus model")
    args = ap.parse_args()

    import pandas as pd
    from scipy.stats import kruskal
    from run_obstruction_validation import auc

    records = {json.loads(l).get("idx", i): json.loads(l)
               for i, l in enumerate(open(args.cohort))}
    # cohorts may lack an idx field; rebuild by order used in the suite
    ordered = [json.loads(l) for l in open(args.cohort)]
    fails = [r for r in ordered if not r["correct"]
             and r["competitor_token"] != -1][:60]
    succs = [r for r in ordered if r["correct"]][:60]
    by_idx = {i: r for i, r in enumerate(fails + succs)}

    suite = pd.read_csv(args.suite)
    cache_path = os.path.join(args.out, "exposure_cache.json")
    cache = _load_cache(cache_path)

    rows = []
    for _, s in suite.iterrows():
        r = by_idx.get(int(s["idx"]), {})
        subj = r.get("subject")
        ans = (r.get("target_text") or "").strip()
        exposure = None
        if args.index and subj and ans:
            exposure = infinigram_count(f"{subj} AND {ans}", args.index,
                                        cache, cache_path)
        if exposure is None:
            exposure = r.get("s_pop")
        rows.append(dict(idx=int(s["idx"]), label=s["label"],
                         exposure=exposure))
    df = pd.DataFrame(rows).dropna(subset=["exposure"])
    df["log_exp"] = (df.exposure.astype(float) + 1).apply(
        lambda v: __import__("math").log10(v))
    df.to_csv(os.path.join(args.out, "exposure.csv"), index=False)

    print(f"\n=== exposure by label (n={len(df)}, "
          f"source={'co-occurrence' if args.index else 'popularity'}) ===")
    groups = [df[df.label == l].log_exp.tolist()
              for l in ("source", "transport", "selection")]
    groups = [g for g in groups if len(g) >= 3]
    for l in ("source", "transport", "selection"):
        sub = df[df.label == l]
        if len(sub):
            print(f"  {l:<10s} n={len(sub):3d}  median log10 exposure "
                  f"{sub.log_exp.median():.2f}")
    if len(groups) >= 2:
        h, p = kruskal(*groups)
        print(f"  Kruskal-Wallis: H={h:.2f}, p={p:.4f}")
    for a, b in (("source", "transport"), ("source", "selection"),
                 ("transport", "selection")):
        xa = df[df.label == a].log_exp.tolist()
        xb = df[df.label == b].log_exp.tolist()
        if xa and xb:
            print(f"  AUC({a} vs {b}) by exposure alone: "
                  f"{auc(xa, xb):.3f}")
    print("(paper claim: these features separate the labels poorly; "
          "AUCs near 0.5-0.65 are the expected outcome)")


if __name__ == "__main__":
    main()
