"""End-to-end smoke test on a tiny random Llama-style model (CPU, seconds).

Checks every certificate and the qualitative behavior of R(S):
  C1  manual forward == HF model logits
  C2  frozen stack reproduces all states & logits
  C3  obstruction residual operator is affine
  C-live  wrapped attention with identity edit == unwrapped model
  R(S) sanity: raising c_plus increases ob; ob ~ 0 when c_plus = realized.
  repairs: all three repair families run and return finite margins.
"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from frozen_cache import Weights, build_cache, certify_frozen, tdla_edge_scores, target_delivery
from obstruction import solve_obstruction
from repairs import (certify_wrappers, repair_transport, repair_selection,
                     repair_source, repair_random_heads, margin)

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=257, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=64,
                  rms_norm_eps=1e-5, rope_theta=10000.0,
                  attn_implementation="eager")
model = LlamaForCausalLM(cfg).float().eval()
W = Weights(model)
ids = torch.randint(0, 257, (12,))

C = build_cache(model, W, ids)                    # C1 inside
err = certify_frozen(W, C)                        # C2
print(f"C1/C2 pass  (frozen reconstruction rel err {err:.2e})")

S = [3, 4]
tok_g = int(C.logits[-1].argsort()[-5])           # some non-top token
tok_c = int(C.logits[-1].argmax())
realized = float(target_delivery(C.resid[-1], W, C, tok_g))

r0 = solve_obstruction(W, C, S, tok_g, c_plus=realized, cg_iters=60)
r1 = solve_obstruction(W, C, S, tok_g, c_plus=realized + 5.0, cg_iters=60)
r2 = solve_obstruction(W, C, S, tok_g, c_plus=realized + 10.0, cg_iters=60)
print(f"C3 pass     (affinity {r1.cert_affine:.2e})")
print(f"ob at realized c+ : {r0.ob:.4f}   (expect ~0)")
print(f"ob at c+ +5       : {r1.ob:.4f}   gap at opt {r1.delivered_gap:+.3f}")
print(f"ob at c+ +10      : {r2.ob:.4f}   (expect > previous)")
assert r0.ob < 1e-2, "ob should vanish when delivery demand equals realized"
assert r2.ob > r1.ob > r0.ob, "ob should grow with the delivery demand"
print(f"depth centroid    : {r1.depth_centroid:.2f}, profile {['%.3f' % p for p in r1.profile]}")

certify_wrappers(model, W, ids)
print("C-live pass")

m0, _ = margin(model, ids, tok_g, tok_c)
mt, _ = repair_transport(model, W, C, ids, tok_g, tok_c, S, k=4)
ms, _ = repair_selection(model, W, C, ids, tok_g, tok_c, k=2)
mr, _ = repair_random_heads(model, W, ids, tok_g, tok_c, k=2)
donor = {l: C.resid[l][S] for l in [1, 2]}        # trivial self-donor
mp, _ = repair_source(model, ids, tok_g, tok_c, S, donor, [1, 2])
assert abs(mp - m0) < 1e-3, "self-donor source patch must be a no-op"
print(f"repairs run: base {m0:+.3f} | transport {mt:+.3f} | "
      f"selection {ms:+.3f} | random {mr:+.3f} | source(self)=no-op OK")

sc = tdla_edge_scores(W, C, tok_g, S)
print(f"tDLA edge scores computed: shape {tuple(sc.shape)}, "
      f"sum {float(sc.sum()):+.4f}")
print("\nALL SMOKE TESTS PASSED")
