"""Experiment 7: controlled-ontogeny toy model.

Trains a tiny Llama on synthetic facts with FULLY CONTROLLED exposure counts
and competitor pressure, then runs the ob(s) diagnostic at checkpoints.

Why this makes the paper's causal case: on real corpora, "long-tail facts
fail as presence" is a correlation with popularity. Here exposure is
assigned, so the claims become interventions:
  P1 exposure -> class: facts in the lowest exposure band fail as presence;
     mid-band failures shift to transport/selection.
  P2 order of emergence: presence dominates early training; transport
     becomes a stable class as accuracy rises; selection appears last
     (paper Sec 6.10, now with controlled data).
  P3 competitor pressure: facts whose object is SHARED by many subjects
     (high competitor pressure) fail disproportionately as selection.

Design: single-token subjects and objects (word-level vocab), facts
"the capital of X is Y", Zipf-assigned exposures, a subset of objects shared
across many subjects (pressure), filler sentences for corpus mass.
Source span = the subject position (single token). Verdicts by the ordered
delivery/content rule calibrated per checkpoint on that checkpoint's own
successes. The tokenizer-free path uses frozen_cache directly on input ids.

Usage:
  python train_toy.py --out results/toy --steps 3000 --device cuda
  (CPU works for the default small config; ~20 min.)
"""
import argparse, json, math, os, random
import torch

from frozen_cache import Weights, build_cache, certify_frozen, target_delivery


# ----------------------------- world ----------------------------------------

def make_world(n_subj=240, n_obj=80, n_shared=12, seed=0):
    """Facts cap(X_i) = Y_j.  The first n_shared objects are 'crowded': many
    subjects map to them (competitor pressure). Exposure ~ Zipf over facts."""
    rng = random.Random(seed)
    TEMPLATE = ["the", "capital", "of", "SUBJ", "is"]
    words = (TEMPLATE[:3] + ["is", "and", "near", "far", "big", "old", "new",
                             "town", "river", "road", "sky", "sun"])
    subj = [f"S{i}" for i in range(n_subj)]
    obj = [f"O{j}" for j in range(n_obj)]
    vocab = ["<pad>"] + sorted(set(words)) + subj + obj
    tid = {w: i for i, w in enumerate(vocab)}

    facts = []
    for i, s in enumerate(subj):
        if i < n_subj // 3:                       # crowded objects
            o = obj[rng.randrange(n_shared)]
        else:
            o = obj[n_shared + rng.randrange(n_obj - n_shared)]
        exposure = max(1, int(200 / (1 + i) ** 0.8))   # Zipf-ish by index
        facts.append(dict(subj=s, obj=o, exposure=exposure,
                          crowded=i < n_subj // 3))
    rng.shuffle(facts)
    return vocab, tid, facts


def make_corpus(vocab, tid, facts, seed=0):
    rng = random.Random(seed)
    sents = []
    for f in facts:
        s = [tid[w] for w in
             ["the", "capital", "of", f["subj"], "is", f["obj"]]]
        sents += [s] * f["exposure"]
    fillers = [w for w in vocab if not (w.startswith("S") or w.startswith("O")
                                        or w == "<pad>")]
    for _ in range(len(sents) // 2):              # filler mass
        sents.append([tid[rng.choice(fillers)] for _ in range(6)])
    rng.shuffle(sents)
    return sents


# ----------------------------- training -------------------------------------

def make_model(vocab_size, device):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=128,
                      intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=32, rms_norm_eps=1e-5,
                      rope_theta=10000.0, attn_implementation="eager")
    return LlamaForCausalLM(cfg).float().to(device)


def train(model, sents, steps, device, batch=64, lr=3e-4, ckpt_steps=(),
          out_dir=".", seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    data = torch.full((len(sents), 6), 0, dtype=torch.long)
    for i, s in enumerate(sents):
        data[i, :len(s)] = torch.tensor(s[:6])
    saved = []
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(sents), (batch,), generator=g)
        x = data[idx].to(device)
        out = model(x, labels=x)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        if step in ckpt_steps:
            p = os.path.join(out_dir, f"ckpt_{step}.pt")
            torch.save(model.state_dict(), p)
            saved.append((step, p))
            print(f"step {step:6d} loss {out.loss.item():.3f}  -> {p}")
    model.eval()
    return saved


# ----------------------------- diagnosis ------------------------------------

@torch.no_grad()
def diagnose(model, tid, facts, device, q=0.10):
    """ob-free light verdicts (delivery/content ordered rule), plus accuracy,
    per fact.  Full ob(s) can be run on the failure subset separately."""
    W = Weights(model)
    rows, caches = [], {}
    for f in facts:
        ids = torch.tensor([tid[w] for w in
                            ["the", "capital", "of", f["subj"], "is"]],
                           device=device)
        C = build_cache(model, W, ids)
        top = int(C.logits[-1].argmax())
        g = tid[f["obj"]]
        ug = ((C.inv_f[-1] * W.ln_f) * W.WU[g]).squeeze()
        lo, hi = W.L // 4, 3 * W.L // 4
        pi = max(float((C.resid[l][3] * ug).sum()) for l in range(lo, hi + 1))
        d = float(target_delivery(C.resid[-1], W, C, g))
        rows.append(dict(subj=f["subj"], obj=f["obj"], exposure=f["exposure"],
                         crowded=f["crowded"], correct=top == g,
                         delivery=d, pi_S=pi))
        caches[f["subj"]] = C
    succ = [r for r in rows if r["correct"]]
    if len(succ) < 5:
        for r in rows:
            r["verdict"] = "correct" if r["correct"] else "presence"
        return rows
    th_d = torch.tensor([r["delivery"] for r in succ]).quantile(q).item()
    th_p = torch.tensor([r["pi_S"] for r in succ]).quantile(q).item()
    for r in rows:
        r["verdict"] = ("correct" if r["correct"] else
                        "selection" if r["delivery"] >= th_d else
                        "transport" if r["pi_S"] >= th_p else "presence")
    return rows


def run(out_dir, steps, device, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    vocab, tid, facts = make_world(seed=seed)
    sents = make_corpus(vocab, tid, facts, seed=seed)
    print(f"world: {len(facts)} facts, corpus {len(sents)} sentences, "
          f"vocab {len(vocab)}")
    model = make_model(len(vocab), device)
    ckpts = sorted({int(steps * f) for f in
                    (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)})
    train(model, sents, steps, device, ckpt_steps=ckpts, out_dir=out_dir,
          seed=seed)

    all_rows = []
    for step, path in [(s, os.path.join(out_dir, f"ckpt_{s}.pt"))
                       for s in ckpts]:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        rows = diagnose(model, tid, facts, device)
        for r in rows:
            r["step"] = step
        all_rows += rows
        acc = sum(r["correct"] for r in rows) / len(rows)
        from collections import Counter
        fails = Counter(r["verdict"] for r in rows if not r["correct"])
        print(f"step {step:6d} acc {acc:.2f} failures {dict(fails)}")

    import csv
    with open(os.path.join(out_dir, "toy_ontogeny.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)
    print(f"wrote {out_dir}/toy_ontogeny.csv")
    return all_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/toy")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.out, args.steps, args.device, args.seed)
