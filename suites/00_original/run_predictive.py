"""Experiment R3: incremental held-out predictive value of affine features.

Targets (never used to construct the ledger):
  rails    whether forcing past the divergence puts the model back on rails
           (classification; from bottleneck.csv)
  effect   magnitude of margin change under the later top-k ablation
           (regression; from obstruction.csv's ablation_effect)

Feature tiers, cumulative:
  T0 black-box   : top logprob, entropy, top-2 margin
  T1 +internal   : pi_S, attention source mass
  T2 +affine      : ob, delivery surplus (gap0), centroid
5-fold CV; ridge for regression (closed form), logistic (few Newton steps)
for classification.  The claim tested: adding affine features improves
held-out prediction beyond confidence + conventional internals.

Usage:
  python run_predictive.py --model <hf> --cohort <jsonl> --out results/<tag>
(reads <out>/obstruction.csv and <out>/bottleneck.csv; recomputes black-box
features with one forward per example)
"""
import argparse, json, os, csv
import torch


def kfold(n, k=5, seed=0):
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return [(torch.cat([idx[j::k] for j in range(k) if j != i]), idx[i::k])
            for i in range(k)]


def ridge(Xtr, ytr, Xte, lam=1.0):
    d = Xtr.shape[1]
    w = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(d), Xtr.T @ ytr)
    return Xte @ w


def logistic(Xtr, ytr, Xte, iters=25, lam=1e-2):
    w = torch.zeros(Xtr.shape[1], dtype=torch.float64)
    for _ in range(iters):                    # Newton
        p = torch.sigmoid(Xtr @ w)
        g = Xtr.T @ (p - ytr) + lam * w
        Hs = Xtr.T @ (Xtr * (p * (1 - p)).unsqueeze(1)) + \
            lam * torch.eye(len(w), dtype=torch.float64)
        w = w - torch.linalg.solve(Hs, g)
    return torch.sigmoid(Xte @ w)


def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import pandas as pd
    from scipy.stats import spearmanr
    from run_obstruction_validation import auc
    from frozen_cache import Weights, build_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    ob = pd.read_csv(f"{args.out}/obstruction.csv")
    bt_path = f"{args.out}/bottleneck.csv"
    bt = pd.read_csv(bt_path) if os.path.exists(bt_path) else None
    records = [json.loads(l) for l in open(args.cohort)]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation="eager").to(args.device).eval()
    W = Weights(model)

    feats = []
    for i, r in enumerate(tqdm(records, desc="black-box features")):
        ids = tok(r["prompt"], return_tensors="pt").input_ids[0].to(args.device)
        C = build_cache(model, W, ids)
        lp = torch.log_softmax(C.logits[-1], -1)
        t2 = torch.topk(C.logits[-1], 2).values
        feats.append(dict(idx=i, logprob=float(lp.max()),
                          entropy=float(-(lp.exp() * lp).sum()),
                          top2=float(t2[0] - t2[1])))
    df = pd.DataFrame(feats).merge(ob, on="idx")
    if bt is not None and "idx" in bt:
        df = df.merge(bt[["idx", "back_on_rails", "diverged"]], on="idx",
                      how="left")

    TIERS = {
        "T0 black-box": ["logprob", "entropy", "top2"],
        "T1 +internal": ["logprob", "entropy", "top2", "pi_S", "attn_mass"],
        "T2 +affine": ["logprob", "entropy", "top2", "pi_S", "attn_mass",
                      "ob", "gap0", "centroid"],
    }
    results = []
    # ---- target 1: back-on-rails (failures with a divergence) --------------
    if bt is not None and "back_on_rails" in df:
        sub = df[(df.verdict != "correct") & df.diverged.fillna(False)
                 ].dropna(subset=["back_on_rails"])
        if len(sub) >= 20:
            y = torch.tensor(sub.back_on_rails.astype(float).values,
                             dtype=torch.float64)
            for tier, cols in TIERS.items():
                X = zscore(torch.tensor(sub[cols].fillna(0).values,
                                        dtype=torch.float64))
                preds = torch.zeros_like(y)
                for tr, te in kfold(len(y)):
                    preds[te] = logistic(X[tr], y[tr], X[te])
                a = auc(preds[y == 1].tolist(), preds[y == 0].tolist())
                results.append(dict(target="rails", tier=tier, n=len(y),
                                    score=a, metric="AUC"))
    # ---- target 2: ablation effect magnitude -------------------------------
    sub = df[df.verdict != "correct"].dropna(subset=["ablation_effect"])
    if len(sub) >= 20:
        y = torch.tensor(sub.ablation_effect.values, dtype=torch.float64)
        for tier, cols in TIERS.items():
            X = zscore(torch.tensor(sub[cols].fillna(0).values,
                                    dtype=torch.float64))
            preds = torch.zeros_like(y)
            for tr, te in kfold(len(y)):
                preds[te] = ridge(X[tr], y[tr], X[te])
            rho = spearmanr(preds.numpy(), y.numpy()).statistic
            results.append(dict(target="ablation_effect", tier=tier,
                                n=len(y), score=rho, metric="Spearman"))

    outp = f"{args.out}/predictive.csv"
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader(); w.writerows(results)
    print("\n=== held-out incremental predictive value ===")
    for r in results:
        print(f"  {r['target']:<16s} {r['tier']:<14s} "
              f"{r['metric']}={r['score']:+.3f}  (n={r['n']})")
    print("claim tested: T2 > T1 > T0 on held-out folds.")


if __name__ == "__main__":
    main()
