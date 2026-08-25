"""Experiment R2: known-circuit calibration benchmark.

Trains a tiny transformer on key-value retrieval where the GROUND-TRUTH
source is known by construction: the answer is the value token paired with
the queried key.  Every sequence also begins with a constant SINK token --
training reliably concentrates attention there (the decoy route: high
attention mass, no answer content).

Per evaluation prompt we score every source POSITION with (a) attention mass
and (b) sheaf transported support (tDLA per column), then check:

  ID@1     does the method's top position equal the true value position?
  ablation zeroing the true value column should destroy the margin;
           zeroing the method's top column measures what the method found.

The compelling comparison: if attention's top position is the sink while the
sheaf's is the true value, 'transported support, not attention mass' is
visible on an object with known structure.  Instrument calibration, not
repair: like testing a microscope on a known specimen.

Usage:  python known_circuit.py --out results/known_circuit --device cuda
CPU-friendly (tiny model, ~2 min).
"""
import argparse, os, random, csv
import torch

from frozen_cache import Weights, build_cache, certify_frozen, tdla_edge_scores
from ranking import ablate_and_measure
from repairs import margin

N_KEYS, N_VALS, N_PAIRS = 20, 20, 3          # pairs per prompt (1 true + 2)
SEQ = 2 + 2 * N_PAIRS + 2                    # SINK k v k v k v Q key ->


def make_model(vocab, device):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=96, intermediate_size=192,
                      num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=SEQ + 2,
                      rms_norm_eps=1e-5, attn_implementation="eager")
    return LlamaForCausalLM(cfg).float().to(device)


def sample_seq(rng, tid):
    keys = rng.sample(range(N_KEYS), N_PAIRS)
    vals = [rng.randrange(N_VALS) for _ in keys]
    qi = rng.randrange(N_PAIRS)
    seq = [tid["SINK"]]
    for k, v in zip(keys, vals):
        seq += [tid[f"K{k}"], tid[f"V{v}"]]
    seq += [tid["Q"], tid[f"K{keys[qi]}"]]
    ans = tid[f"V{vals[qi]}"]
    val_pos = 2 + 2 * qi                       # position of the true value
    return seq, ans, val_pos


def run(out_dir="results/known_circuit", steps=2500, n_eval=150,
        device="cpu", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    vocab = ["<pad>", "SINK", "Q"] + [f"K{i}" for i in range(N_KEYS)] + \
            [f"V{j}" for j in range(N_VALS)]
    tid = {w: i for i, w in enumerate(vocab)}
    model = make_model(len(vocab), device)

    # ---- train -------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    try:
        from tqdm import trange
        it = trange(steps, desc="train known-circuit")
    except ImportError:
        it = range(steps)
    with torch.enable_grad():
        for step in it:
            batch, answers = [], []
            for _ in range(64):
                s, a, _ = sample_seq(rng, tid)
                batch.append(s + [a])
                answers.append(a)
            x = torch.tensor(batch, device=device)
            labels = torch.full_like(x, -100)
            labels[:, -1] = x[:, -1]           # supervise only the answer
            loss = model(x, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
            if hasattr(it, "set_postfix"):
                it.set_postfix(loss=f"{loss.item():.3f}")
    model.eval()

    # ---- evaluate source identification -------------------------------------
    W = Weights(model)
    rows = []
    n_correct = 0
    for i in range(n_eval):
        s, ans, val_pos = sample_seq(rng, tid)
        ids = torch.tensor(s, device=device)
        C = build_cache(model, W, ids)
        certify_frozen(W, C)
        top = int(C.logits[-1].argmax())
        if top != ans:
            continue                            # calibrate on solved prompts
        n_correct += 1
        comp = int(C.logits[-1].argsort()[-2])
        cols = list(range(len(s) - 1))          # candidate source columns
        # attention mass per column (all layers/heads):
        attn = torch.stack([C.layers[l].A[:, -1, :].sum(0)
                            for l in range(W.L)]).sum(0)[:len(s) - 1]
        # sheaf transported support per column:
        sheaf = tdla_edge_scores(W, C, ans, cols,
                                 tok_c=comp).sum(0).sum(0)   # [|cols|]
        top_attn, top_sheaf = int(attn.argmax()), int(sheaf.argmax())
        m0, _ = margin(model, ids, ans, comp)
        drop = lambda col: m0 - ablate_and_measure(
            model, W, ids, ans, comp, [col],
            [(l, h) for l in range(W.L) for h in range(W.H)])
        rows.append(dict(
            idx=i, val_pos=val_pos, top_attn=top_attn, top_sheaf=top_sheaf,
            attn_hit=top_attn == val_pos, sheaf_hit=top_sheaf == val_pos,
            attn_top_is_sink=top_attn == 0,
            drop_true=drop(val_pos), drop_attn_top=drop(top_attn),
            drop_sheaf_top=drop(top_sheaf)))

    with open(os.path.join(out_dir, "known_circuit.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    summary = dict(
        n=n, task_acc=n_correct / n_eval,
        id1_sheaf=sum(r["sheaf_hit"] for r in rows) / n,
        id1_attn=sum(r["attn_hit"] for r in rows) / n,
        attn_on_sink=sum(r["attn_top_is_sink"] for r in rows) / n,
        med_drop_true=sorted(r["drop_true"] for r in rows)[n // 2],
        med_drop_sheaf_top=sorted(r["drop_sheaf_top"] for r in rows)[n // 2],
        med_drop_attn_top=sorted(r["drop_attn_top"] for r in rows)[n // 2])
    print("\n=== known-circuit calibration ===")
    for k, v in summary.items():
        print(f"  {k:<20s} {v:.3f}" if isinstance(v, float) else
              f"  {k:<20s} {v}")
    print("prediction: id1_sheaf >> id1_attn; attention's top column is the "
          "sink; ablating the sheaf's top column destroys the margin.")
    return summary, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/known_circuit")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.out, a.steps, device=a.device, seed=a.seed)
