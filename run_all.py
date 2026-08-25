"""Orchestrator: full multi-model analysis.

Per model: build cohort (if missing) -> exp 1 (obstruction) -> exp 4 (AUC)
-> exp 2 (repairs) -> exp 3 (sensitivity) -> exp 5 (bottleneck).
Each stage is skipped if its output already exists (resume-friendly);
stage logs land in results/<tag>/logs/.  Final step: cross-model aggregation.

Model scope = the frozen contract: Llama + Qwen2.5 families (RMSNorm with
weights, full RoPE, gated SiLU MLP).  Qwen3 (QK-norm), Gemma-2 (softcap),
OLMo (non-parametric LN), Pythia (parallel blocks) need the paper's reduced
tiers and are not run here.

fp32 memory: 1B ~ 6 GB, 3B ~ 14 GB, 8B ~ 34 GB + activations.  Run 8B on an
80 GB card, or split MODELS across GPUs (one process per GPU).

Usage:
  python run_all.py                        # everything, default model list
  python run_all.py --models Llama-3.2-1B  # subset by substring
  python run_all.py --stages cohort,ob,auc # subset of stages
"""
import argparse, os, subprocess, sys, time

MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "allenai/OLMo-7B-hf",     # LayerNorm family; center-aware contract
]

STAGES = ("cohort", "ob", "auc", "repair", "sens", "bottleneck")


def tag(m):
    return m.split("/")[-1].replace(".", "").lower()


def run(cmd, log):
    print(f"  $ {' '.join(cmd)}\n    -> {log}")
    with open(log, "w") as f:
        t0 = time.time()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
    status = "ok" if p.returncode == 0 else f"FAILED (rc={p.returncode})"
    print(f"    {status} in {dt/60:.1f} min")
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--standin", action="store_true",
                    help="stand-in verdicts for cohorts lacking pipeline labels")
    args = ap.parse_args()
    stages = args.stages.split(",")
    models = [m for m in MODELS if args.models.lower() in m.lower()]
    py = sys.executable

    for m in models:
        t = tag(m)
        cohort = f"cohorts/{t}_parametric.jsonl"
        out = f"results/{t}"
        logs = f"{out}/logs"
        os.makedirs(logs, exist_ok=True)
        print(f"\n=== {m} ===")

        if "cohort" in stages and not os.path.exists(cohort):
            cmd = [py, "build_cohort.py", "--model", m, "--popqa",
                   "--out", cohort, "--device", args.device]
            if args.standin:
                cmd.append("--standin")
            if not run(cmd, f"{logs}/cohort.log"):
                continue
        if not os.path.exists(cohort):
            print(f"  no cohort at {cohort}; skipping model")
            continue

        plan = [
            ("ob", f"{out}/obstruction.csv",
             [py, "run_obstruction_validation.py", "--model", m,
              "--cohort", cohort, "--out", out, "--run_ablation",
              "--device", args.device]),
            ("auc", f"{out}/auc_battery.csv",
             [py, "run_auc_battery.py", "--model", m, "--cohort", cohort,
              "--obcsv", f"{out}/obstruction.csv", "--out", out,
              "--device", args.device]),
            ("repair", f"{out}/repair_matrix.csv",
             [py, "run_repair_matrix.py", "--model", m, "--cohort", cohort,
              "--out", out, "--device", args.device]),
            ("sens", f"{out}/sensitivity.csv",
             [py, "run_sensitivity.py", "--model", m, "--cohort", cohort,
              "--out", out, "--device", args.device]),
            ("bottleneck", f"{out}/bottleneck.csv",
             [py, "run_bottleneck.py", "--model", m, "--cohort", cohort,
              "--out", out, "--device", args.device]),
        ]
        for name, artifact, cmd in plan:
            if name not in stages:
                continue
            if os.path.exists(artifact):
                print(f"  [{name}] exists, skipping")
                continue
            run(cmd, f"{logs}/{name}.log")

    print("\n=== aggregation ===")
    subprocess.run([py, "aggregate_results.py"])


if __name__ == "__main__":
    main()
