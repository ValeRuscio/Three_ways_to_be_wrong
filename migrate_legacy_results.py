"""One-time migration of pre-rename results/cohorts to the paper's
terminology: verdict 'presence' -> 'source'; CSV columns sheaf_* -> affine_*
and presence_patch* -> source_patch*.  Idempotent; run from the repo root.
Usage: python migrate_legacy_results.py
"""
import csv, glob, json, os

for path in glob.glob("results/**/*.csv", recursive=True):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        continue
    ren = {k: k.replace("sheaf_", "affine_")
              .replace("presence_patch", "source_patch")
           for k in rows[0]}
    changed = any(a != b for a, b in ren.items())
    for r in rows:
        for col in ("verdict", "label"):
            if r.get(col) == "presence":
                r[col] = "source"
                changed = True
    if changed:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[ren[k] for k in rows[0]])
            w.writeheader()
            w.writerows([{ren[k]: v for k, v in r.items()} for r in rows])
        print("migrated", path)

for path in glob.glob("cohorts/*.jsonl"):
    out, changed = [], False
    for line in open(path):
        r = json.loads(line)
        if r.get("verdict") == "presence":
            r["verdict"] = "source"
            changed = True
        out.append(r)
    if changed:
        with open(path, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print("migrated", path)
print("done")
