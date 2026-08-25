"""Experiment 5: first-token bottleneck, resolved by verdict class.

Answers the reviewer's questions directly:
  - at which answer token does generation diverge from the target?
  - forced past the divergence, does the model complete the target exactly
    (back-on-rails)?
  - how do divergence depth and back-on-rails rate break down by verdict?

The paper's prediction: bottleneck (divergence at token 0 + back-on-rails)
should concentrate in selection and transport failures -- the answer exists
and initiation fails -- and be rare in presence failures, where there is
nothing to get back onto the rails of.

Cohort needs `target_text` (the full canonical answer string) per record.

Usage:
  python run_bottleneck.py --model <hf> --cohort <jsonl> --out <dir>
"""
import argparse, json, os, csv
import torch


@torch.no_grad()
def forced_trajectory(model, tok, prompt, target_text, device):
    """Greedy check along the forced target; divergence step + back-on-rails."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    tgt = tok(target_text, add_special_tokens=False).input_ids
    cur, div = ids, None

    def same_word(a, b):
        # word-level divergence: token-id mismatches that decode to the same
        # word modulo whitespace/case (leading-space variants, casing) are NOT
        # divergences; raw id comparison inflates div@0 on correct examples.
        return (a == b or
                tok.decode([a]).strip().lower() == tok.decode([b]).strip().lower())

    for step, t in enumerate(tgt):
        top = int(model(cur).logits[0, -1].argmax())
        if not same_word(top, t) and div is None:
            div = step
        cur = torch.cat([cur, torch.tensor([[t]], device=device)], dim=1)
    if div is None:
        return dict(diverged=False, div_step=-1, back_on_rails=None,
                    n_target_tokens=len(tgt))
    # force through the divergence token, then require exact greedy completion
    cur = torch.cat([ids, torch.tensor([tgt[:div + 1]], device=device)], dim=1)
    rails = True
    for t in tgt[div + 1:]:
        if not same_word(int(model(cur).logits[0, -1].argmax()), t):
            rails = False
            break
        cur = torch.cat([cur, torch.tensor([[t]], device=device)], dim=1)
    return dict(diverged=True, div_step=div,
                back_on_rails=rails if div + 1 < len(tgt) else True,
                n_target_tokens=len(tgt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation="eager").to(args.device).eval()

    records = [json.loads(l) for l in open(args.cohort)
               if json.loads(l).get("target_text")]
    rows = []
    for i, r in enumerate(records):
        t = forced_trajectory(model, tok, r["prompt"], r["target_text"],
                              args.device)
        t.update(idx=i, verdict=r["verdict"])
        rows.append(t)
        print(f"[{i:4d}] {r['verdict']:<9s} div={t['div_step']:3d} "
              f"rails={t['back_on_rails']}")

    with open(os.path.join(args.out, "bottleneck.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'verdict':<10s} {'n':>4s} {'div@0':>6s} {'rails':>6s} "
          f"{'med div':>8s}")
    for v in ("presence", "transport", "selection", "correct"):
        sub = [r for r in rows if r["verdict"] == v and r["diverged"]]
        if not sub:
            continue
        d0 = sum(r["div_step"] == 0 for r in sub) / len(sub)
        rails = sum(bool(r["back_on_rails"]) for r in sub) / len(sub)
        med = sorted(r["div_step"] for r in sub)[len(sub) // 2]
        print(f"{v:<10s} {len(sub):4d} {d0:6.2f} {rails:6.2f} {med:8d}")
    print("\nPrediction: div@0 & rails concentrate in selection/transport, "
          "not presence.")


if __name__ == "__main__":
    main()
