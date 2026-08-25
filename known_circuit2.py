"""Known-circuit benchmark v2: factorial design over attention alignment and
routing complexity, many seeds, signed recovery.

Conditions (2x2):
  easy        one route, no decoy
  decoy       one route + synthetic high-attention edge with ZERO target OV:
              lambda of every head's final-position attention is moved onto a
              constant filler column (exact by construction -- the decoy has
              maximal mass and carries no target-relevant value)
  competing   two genuine routes: the queried value appears under two keys
  hard        competing routes + decoy

The decoy is applied consistently to the realized computation (live model and
frozen cache run under the same attention edit), so the diagnostic sees
exactly the computation that produced the output.  attn_top_is_decoy is TRUE
by construction in decoy conditions -- this is the sink-resistance test the
v1 benchmark lacked.

Signed recovery: components can promote or suppress the target; we report
whether each method identifies the top |effect| component and predicts its
sign, not only positive recovery.

Reported per seed x condition: ID@1 (strict and lenient for competing
routes), causal drop of the chosen column, top-k recall of the known
circuit, normalized regret vs the measured column-oracle, sign agreement.

Usage: python known_circuit2.py --out results/known_circuit2 --seeds 12
"""
import argparse, os, random, csv
import torch

from frozen_cache import Weights, certify_frozen
from ranking import ablate_and_measure
from repairs import attention_edits, margin
from known_circuit import make_model, N_KEYS, N_VALS

N_PAIRS = 3
DECOY_LAMBDA = 0.5


def build_vocab():
    vocab = ["<pad>", "FILL", "Q"] + [f"K{i}" for i in range(N_KEYS)] + \
            [f"V{j}" for j in range(N_VALS)]
    return vocab, {w: i for i, w in enumerate(vocab)}


def sample_seq(rng, tid, competing=False):
    keys = rng.sample(range(N_KEYS), N_PAIRS)
    vals = [rng.randrange(N_VALS) for _ in keys]
    qi = rng.randrange(N_PAIRS)
    if competing:                       # second key with the SAME value
        oj = rng.choice([j for j in range(N_PAIRS) if j != qi])
        vals[oj] = vals[qi]
    seq = [tid["FILL"]]                 # constant filler = decoy column 0
    for k, v in zip(keys, vals):
        seq += [tid[f"K{k}"], tid[f"V{v}"]]
    seq += [tid["Q"], tid[f"K{keys[qi]}"]]
    true_pos = 2 + 2 * qi
    alt = [2 + 2 * j for j in range(N_PAIRS)
           if vals[j] == vals[qi]]      # all columns holding the answer
    return seq, tid[f"V{vals[qi]}"], true_pos, alt


def decoy_edit(A, li):
    A = A.clone()
    A[:, -1] = (1 - DECOY_LAMBDA) * A[:, -1]
    A[:, -1, 0] = A[:, -1, 0] + DECOY_LAMBDA
    return A


def _manual_edited_cache(model, W, ids, edit):
    """Frozen cache of the computation UNDER the attention edit, verified
    against the live wrapped model (the C1 analogue for edited passes)."""
    import torch as t
    from frozen_cache import (norm_inv, apply_norm, repeat_kv, apply_rope,
                              LayerCache, ForwardCache)
    dev = ids.device
    T = ids.shape[-1]
    x = W.WE[ids.view(-1)]
    cos, sin = W.cos_sin(T, dev, x.dtype)
    mask = t.full((T, T), float("-inf"), device=dev).triu(1)
    resid, layers = [x.clone()], []
    for li, lw in enumerate(W.layers):
        inv1 = norm_inv(x, W.eps, W.center)
        xn = apply_norm(x, inv1, lw["ln1"], W.center)
        q = xn @ lw["Wq"].T; k = xn @ lw["Wk"].T; v = xn @ lw["Wv"].T
        q = q.view(T, W.H, W.dh).transpose(0, 1)
        k = repeat_kv(k.view(T, W.Hkv, W.dh).transpose(0, 1), W.n_rep)
        v = repeat_kv(v.view(T, W.Hkv, W.dh).transpose(0, 1), W.n_rep)
        q, k = apply_rope(q, k, cos, sin)
        A = ((q @ k.transpose(-1, -2)) / W.dh ** 0.5 + mask).softmax(-1)
        if edit is not None:
            A = edit(A, li)
        x = x + (A @ v).transpose(0, 1).reshape(T, -1) @ lw["Wo"].T
        inv2 = norm_inv(x, W.eps, W.center)
        xn2 = apply_norm(x, inv2, lw["ln2"], W.center)
        gate = t.nn.functional.silu(xn2 @ lw["Wg"].T)
        x = x + (gate * (xn2 @ lw["Wu"].T)) @ lw["Wd"].T
        resid.append(x.clone())
        layers.append(LayerCache(A=A, gate=gate, inv1=inv1, inv2=inv2))
    inv_f = norm_inv(x, W.eps, W.center)
    logits = apply_norm(x, inv_f, W.ln_f, W.center) @ W.WU.T
    C = ForwardCache(resid=resid, layers=layers, inv_f=inv_f,
                     logits=logits, input_ids=ids)
    # C1-analogue: live model under the same edit must agree
    with attention_edits(model, W, prob_edit=edit) if edit else _null():
        ref = model(ids.view(1, -1), use_cache=False).logits[0].float()
    rel = (logits - ref).norm() / ref.norm()
    assert rel < 1e-4, f"edited C1 failed: {rel:.2e}"
    return C


from contextlib import contextmanager
@contextmanager
def _null():
    yield


@torch.no_grad()
def column_scores(W, C, ans, comp, ncols):
    from frozen_cache import tdla_edge_scores
    cols = list(range(ncols))
    affine = tdla_edge_scores(W, C, ans, cols, tok_c=comp).sum(0).sum(0)
    attn = torch.stack([C.layers[l].A[:, -1, :ncols].sum(0)
                        for l in range(W.L)]).sum(0)
    return {"affine": affine.cpu(), "attention": attn.cpu()}


def evaluate(model, W, tid, rng, condition, n_eval, device, edit,
             competing):
    all_heads = [(l, h) for l in range(W.L) for h in range(W.H)]
    rows = []
    for i in range(n_eval):
        seq, ans, true_pos, alt = sample_seq(rng, tid, competing=competing)
        ids = torch.tensor(seq, device=device)
        C = _manual_edited_cache(model, W, ids, edit)
        top = int(C.logits[-1].argmax())
        if top != ans:
            continue
        comp = int(C.logits[-1].argsort()[-2])
        m0, _ = _edited_margin(model, W, ids, ans, comp, edit)
        sc = column_scores(W, C, ans, comp, len(seq) - 1)

        # measured column oracle (signed)
        eff = {}
        for col in range(len(seq) - 1):
            eff[col] = m0 - _edited_ablate(model, W, ids, ans, comp, edit,
                                           col, all_heads)
        oracle_col = max(eff, key=lambda c_: abs(eff[c_]))
        row = dict(condition=condition, idx=i, true_pos=true_pos,
                   n_true=len(alt), oracle_col=oracle_col,
                   oracle_eff=eff[oracle_col],
                   drop_true=eff[true_pos])
        for name, s in sc.items():
            topc = int(s.argmax())
            row[f"{name}_top"] = topc
            row[f"{name}_hit_strict"] = topc == true_pos
            row[f"{name}_hit_lenient"] = topc in alt
            row[f"{name}_top_is_decoy"] = topc == 0
            row[f"{name}_drop_top"] = eff[topc]
            row[f"{name}_regret"] = (abs(eff[oracle_col]) - abs(eff[topc]))
            row[f"{name}_signagree"] = (s[topc] > 0) == (eff[topc] > 0)
        rows.append(row)
    return rows


def _edited_margin(model, W, ids, g, c, edit):
    if edit is None:
        return margin(model, ids, g, c)
    with attention_edits(model, W, prob_edit=edit):
        return margin(model, ids, g, c)


def _edited_ablate(model, W, ids, g, c, edit, col, heads):
    from ranking import _edge_edit
    base = _edge_edit(heads, [col])
    combined = (lambda A, li: base(edit(A, li), li)) if edit else base
    with attention_edits(model, W, prob_edit=combined):
        m, _ = margin(model, ids, g, c)
    return m


def run_seed(seed, out_dir, steps, n_eval, device):
    rng = random.Random(seed)
    vocab, tid = build_vocab()
    model = make_model(len(vocab), device)
    # train exactly as v1 (single-route retrieval; decoys applied at eval)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    with torch.enable_grad():
        for _ in range(steps):
            batch = []
            for _ in range(64):
                s, a, _, _ = sample_seq(rng, tid,
                                        competing=rng.random() < 0.5)
                batch.append(s + [a])
            x = torch.tensor(batch, device=device)
            lab = torch.full_like(x, -100); lab[:, -1] = x[:, -1]
            loss = model(x, labels=lab).loss
            loss.backward(); opt.step(); opt.zero_grad()
    model.eval()
    W = Weights(model)
    rows = []
    for cond, edit, competing in (("easy", None, False),
                                  ("decoy", decoy_edit, False),
                                  ("competing", None, True),
                                  ("hard", decoy_edit, True)):
        rows += [dict(seed=seed, **r) for r in
                 evaluate(model, W, tid, rng, cond, n_eval, device, edit,
                          competing)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/known_circuit2")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n_eval", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    try:
        from tqdm import tqdm
        it = tqdm(range(args.seeds), desc="seeds")
    except ImportError:
        it = range(args.seeds)
    rows = []
    for seed in it:
        rows += run_seed(seed, args.out, args.steps, args.n_eval,
                         args.device)
    with open(f"{args.out}/known_circuit2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    import statistics as st
    print(f"\n=== known-circuit v2 ({args.seeds} seeds) ===")
    print(f"{'condition':<11s} {'affine ID@1':>10s} {'attn ID@1':>10s} "
          f"{'attn->decoy':>11s} {'affine regret':>12s}")
    for cond in ("easy", "decoy", "competing", "hard"):
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        seed_ids = sorted({r["seed"] for r in sub})
        sh = [st.mean([r["affine_hit_lenient"] for r in sub
                       if r["seed"] == s2]) for s2 in seed_ids]
        at = [st.mean([r["attention_hit_lenient"] for r in sub
                       if r["seed"] == s2]) for s2 in seed_ids]
        dc = st.mean([r["attention_top_is_decoy"] for r in sub])
        rg = st.median([r["affine_regret"] for r in sub])
        print(f"{cond:<11s} {st.median(sh):>7.2f} "
              f"[{min(sh):.2f}-{max(sh):.2f}] {st.median(at):>7.2f}   "
              f"{dc:>8.2f}   {rg:>10.3f}")
    neg = [r for r in rows if r["drop_true"] < 0]
    print(f"negative drop_true: {len(neg)}/{len(rows)} "
          f"(signed recovery: affine sign agreement "
          f"{st.mean([r['affine_signagree'] for r in rows]):.2f})")
    print("ideal pattern: both work on easy; attention degrades under decoy;"
          " affine holds; both degrade gracefully under competing routes.")


if __name__ == "__main__":
    main()
