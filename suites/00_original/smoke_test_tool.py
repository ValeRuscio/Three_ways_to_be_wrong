"""Smoke test for the tool-paper battery (R1-R5 components), tiny model, CPU."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from frozen_cache import Weights, build_cache, certify_frozen
from repairs import certify_wrappers, margin
from ranking import (score_candidates, ground_truth, candidate_subset,
                     ranking_metrics, cumulative_curves, ablate_and_measure,
                     RANKERS)

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=257, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=64,
                  rms_norm_eps=1e-5, attn_implementation="eager")
model = LlamaForCausalLM(cfg).float().eval()
W = Weights(model)
ids = torch.randint(0, 257, (12,))
C = build_cache(model, W, ids)
certify_frozen(W, C)
certify_wrappers(model, W, ids)
g = int(C.logits[-1].argsort()[-5]); c = int(C.logits[-1].argmax())
S = [3, 4]
m0, _ = margin(model, ids, g, c)

scores = score_candidates(model, W, C, ids, g, c, S)
assert set(scores) == set(RANKERS) and scores["affine"].shape == (4, 4)
print("scores OK:", {k: f"{v.abs().sum():.3f}" for k, v in scores.items()})

cands = [(l, h) for l in range(4) for h in range(4)]     # full measurement
eff = ground_truth(model, W, ids, g, c, S, cands, m0)
assert len(eff) == 16 and all(v == v for v in eff.values())
print(f"ground truth OK: {min(eff.values()):+.3f} .. {max(eff.values()):+.3f}")

met = ranking_metrics(scores, eff)
assert all(f"{n}_rho" in met for n in RANKERS)
print("metrics OK:", {n: f"{met[f'{n}_rho']:+.2f}" for n in RANKERS})

sub = candidate_subset(scores, top=3, n_random=4)
assert 3 <= len(sub) <= 16
curves = cumulative_curves(model, W, ids, g, c, S, scores, m0,
                           effects=eff, ks=(1, 4))
assert "oracle_drop@4" in curves and "affine_drop@4" in curves
print(f"curves OK: oracle@4 {curves['oracle_drop@4']:+.3f}, "
      f"affine@4 {curves['affine_drop@4']:+.3f}, "
      f"random@4 {curves['random_drop@4']:+.3f}")

# joint-vs-single consistency: ablating one candidate two ways must agree
one = list(eff)[0]
m_joint = ablate_and_measure(model, W, ids, g, c, S, [one])
assert abs((m0 - m_joint) - eff[one]) < 1e-5
print("joint/single consistency OK")

# known-circuit: a very short training run must complete end-to-end
from known_circuit import run as kc_run
summary, rows = kc_run("/tmp/kc_smoke", steps=250, n_eval=40, device="cpu")
assert summary["n"] >= 1
print("known-circuit pipeline OK")

print("\nALL TOOL-PAPER SMOKE TESTS PASSED")
