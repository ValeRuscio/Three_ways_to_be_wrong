"""Experiment 4: three-tier AUC battery for failure-class discrimination.

Tiers:
  black-box : top-token logprob, entropy, top-2 logit margin       (paper: 0.59)
  internal  : pi_S (patch content), tau (transported support)      (paper: 0.98)
  sheaf     : ob(s), depth centroid, ob_norm                       (new)

Tasks: presence-vs-transport (the pair with opposite remedies), and each
class vs rest.  If the sheaf tier matches the internal tier WITHOUT the
calibrated tau thresholds, the obstruction is a self-contained diagnostic.

Single-feature AUCs plus a two-feature Fisher discriminant (closed form, no
sklearn).  Expects the per-example CSV from run_obstruction_validation.py,
augmented here with black-box features recomputed from the cached logits.

Usage:
  python run_auc_battery.py --model <hf> --cohort <jsonl> --obcsv <obstruction.csv> --out <dir>
"""
import argparse, json, os, csv
import torch

from frozen_cache import Weights, build_cache
from run_obstruction_validation import auc

TASKS = [("presence", "transport"), ("presence", None),
         ("transport", None), ("selection", None)]

TIERS = {
    "black-box": ["logprob", "entropy", "top2margin"],
    "internal":  ["pi_S", "tau"],
    "sheaf":     ["ob", "centroid", "ob_norm"],
}


def fisher_auc(rows, feats, pos_label, neg_labels):
    """AUC of a 2-class Fisher discriminant on the given features."""
    def X(labels):
        return torch.tensor([[r[f] for f in feats] for r in rows
                             if r["verdict"] in labels and
                             all(r.get(f) is not None for f in feats)],
                            dtype=torch.float64)
    Xp, Xn = X({pos_label}), X(neg_labels)
    if len(Xp) < 2 or len(Xn) < 2:
        return float("nan")
    mu_p, mu_n = Xp.mean(0), Xn.mean(0)
    Sw = torch.cov(Xp.T) * (len(Xp) - 1) + torch.cov(Xn.T) * (len(Xn) - 1)
    Sw = Sw + 1e-6 * torch.eye(len(feats), dtype=torch.float64)
    w = torch.linalg.solve(Sw, (mu_p - mu_n))
    return auc((Xp @ w).tolist(), (Xn @ w).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--obcsv", required=True)
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
    obrows = {int(r["idx"]): r for r in csv.DictReader(open(args.obcsv))}

    rows = []
    for i, r in enumerate(records):
        ids = tok(r["prompt"], return_tensors="pt").input_ids[0].to(args.device)
        C = build_cache(model, W, ids)
        logp = torch.log_softmax(C.logits[-1], -1)
        top2 = torch.topk(C.logits[-1], 2).values
        o = obrows.get(i, {})
        fl = lambda k: None if o.get(k) in (None, "", "nan") else float(o[k])
        rows.append(dict(
            verdict=r["verdict"],
            logprob=float(logp.max()),
            entropy=float(-(logp.exp() * logp).sum()),
            top2margin=float(top2[0] - top2[1]),
            pi_S=fl("pi_S"), tau=fl("tau"),
            ob=fl("ob"), centroid=fl("centroid"), ob_norm=fl("ob_norm")))

    print(f"{'task':<24s}" + "".join(f"{t:>12s}" for t in TIERS))
    table = []
    for pos, neg in TASKS:
        negs = {neg} if neg else \
            ({"presence", "transport", "selection", "correct"} - {pos})
        name = f"{pos} vs {neg or 'rest'}"
        line, rec = f"{name:<24s}", dict(task=name)
        for tier, feats in TIERS.items():
            # best single feature, and the Fisher combination
            singles = [auc([r[f] for r in rows if r["verdict"] == pos
                            and r.get(f) is not None],
                           [r[f] for r in rows if r["verdict"] in negs
                            and r.get(f) is not None]) for f in feats]
            a = max([s for s in singles if s == s] +
                    [fisher_auc(rows, feats, pos, negs)] or [float("nan")],
                    default=float("nan"))
            line += f"{a:12.3f}"
            rec[tier] = a
        print(line)
        table.append(rec)

    with open(os.path.join(args.out, "auc_battery.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)


if __name__ == "__main__":
    main()
