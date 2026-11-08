"""Cross-model aggregation: the tables and figures for the paper.

Reads results/<tag>/{obstruction,auc_battery,repair_matrix,sensitivity,
bottleneck}.csv for every model and produces in results/aggregate/:

  T1_verdict_shares.csv   verdict decomposition per model + equivalence bound
  T2_ob_separation.csv    R(S) class medians and AUCs per model
  T3_auc_battery.csv      pooled three-tier AUC table
  T4_repair_matrix.csv    pooled repair matrix (mean margin change, flip rate)
  T5_bottleneck.csv       div@0 and back-on-rails by verdict, pooled
  F1_shares.png           stacked verdict shares
  F2_ob_by_class.png      ob distributions by class across models
"""
import glob, os
import pandas as pd

from stats_utils import homogeneity_equivalence
from run_obstruction_validation import auc

OUT = "results/aggregate"
os.makedirs(OUT, exist_ok=True)


def per_model(pattern):
    for p in sorted(glob.glob(f"results/*/{pattern}")):
        model = p.split(os.sep)[-2]
        if model != "aggregate":
            yield model, pd.read_csv(p)


def main():
    # ---- T1: verdict shares + equivalence -------------------------------
    shares, labels = [], {}
    for m, df in per_model("obstruction.csv"):
        f = df[df.verdict != "correct"]
        if not len(f):
            continue
        # homogeneity/equivalence is a within-task (parametric) claim:
        # comparing task families to each other trivially gives TV ~ 1.
        if not m.endswith(("_extraction", "_twohop")) and "toy" not in m:
            labels[m] = f.verdict.tolist()
        s = f.verdict.value_counts(normalize=True)
        shares.append(dict(model=m, n=len(f),
                           source=s.get("source", 0.0),
                           transport=s.get("transport", 0.0),
                           selection=s.get("selection", 0.0)))
    t1 = pd.DataFrame(shares)
    if len(labels) >= 2:
        obs, bound, pairs = homogeneity_equivalence(labels)
        print(f"[T1] observed max pairwise TV {obs:.3f}; "
              f"95% equivalence bound {bound:.3f}")
        t1.attrs = {}
        with open(f"{OUT}/T1_equivalence.txt", "w") as fh:
            fh.write(f"max_TV_observed {obs:.4f}\nTV_95_bound {bound:.4f}\n")
    t1.to_csv(f"{OUT}/T1_verdict_shares.csv", index=False)

    # ---- T2: ob separation per model ------------------------------------
    rows = []
    for m, df in per_model("obstruction.csv"):
        g = lambda v, k="ob": df[df.verdict == v][k].dropna().tolist()
        rows.append(dict(
            model=m,
            med_source=pd.Series(g("source")).median(),
            med_transport=pd.Series(g("transport")).median(),
            med_selection=pd.Series(g("selection")).median(),
            med_correct=pd.Series(g("correct")).median(),
            auc_pt_vs_sc=auc(g("source") + g("transport"),
                             g("selection") + g("correct")),
            auc_pres_vs_trans_centroid=auc(g("transport", "centroid"),
                                           g("source", "centroid"))))
    pd.DataFrame(rows).to_csv(f"{OUT}/T2_ob_separation.csv", index=False)

    # ---- T3: pooled AUC battery -----------------------------------------
    frames = [df.assign(model=m) for m, df in per_model("auc_battery.csv")]
    if frames:
        t3 = pd.concat(frames)
        pooled = t3.groupby("task")[["black-box", "internal", "affine"]].mean()
        pooled.to_csv(f"{OUT}/T3_auc_battery.csv")
        print("[T3] pooled AUC battery:\n", pooled.round(3))

    # ---- T4: pooled repair matrix ---------------------------------------
    frames = [df.assign(model=m) for m, df in per_model("repair_matrix.csv")]
    if frames:
        t4 = pd.concat(frames)
        cols = [c for c in ("source_patch", "transport_edges",
                            "transport_force", "selection_demoters",
                            "random_heads") if c in t4]
        mat = t4.groupby("verdict")[cols].mean()
        flip = t4.groupby("verdict")[[c + "_flip" for c in cols
                                      if c + "_flip" in t4]].mean()
        mat.to_csv(f"{OUT}/T4_repair_matrix.csv")
        flip.to_csv(f"{OUT}/T4_repair_flips.csv")
        print("[T4] pooled repair matrix:\n", mat.round(2))

    # ---- T5: bottleneck ---------------------------------------------------
    frames = [df.assign(model=m) for m, df in per_model("bottleneck.csv")]
    if frames:
        t5 = pd.concat(frames)
        t5 = t5[t5.diverged == True]  # noqa: E712
        agg = t5.groupby("verdict").agg(
            n=("div_step", "size"),
            div0=("div_step", lambda s: (s == 0).mean()),
            rails=("back_on_rails", "mean"),
            med_div=("div_step", "median"))
        agg.to_csv(f"{OUT}/T5_bottleneck.csv")
        print("[T5] bottleneck by verdict:\n", agg.round(2))

    # ---- figures ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(t1):
            ax = t1.set_index("model")[
                ["source", "transport", "selection"]].plot(
                kind="bar", stacked=True, figsize=(9, 3.2))
            ax.set_ylabel("share of failures")
            plt.tight_layout(); plt.savefig(f"{OUT}/F1_shares.png", dpi=150)
        frames = [df.assign(model=m) for m, df in per_model("obstruction.csv")]
        if frames:
            allob = pd.concat(frames)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            order = ["source", "transport", "selection", "correct"]
            data = [allob[allob.verdict == v].ob.dropna() for v in order]
            ax.boxplot([d for d in data if len(d)],
                       labels=[v for v, d in zip(order, data) if len(d)])
            ax.set_ylabel("R(S)"); ax.set_title("pooled across models")
            plt.tight_layout(); plt.savefig(f"{OUT}/F2_ob_by_class.png", dpi=150)
        print(f"[figs] written to {OUT}/")
    except ImportError:
        print("matplotlib not available; tables only")


if __name__ == "__main__":
    main()
