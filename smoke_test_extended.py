"""Smoke test for experiments 3-6 on a tiny random model (CPU, <1 min).
Run with the same python env as smoke_test.py.
"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from frozen_cache import Weights, build_cache, certify_frozen, target_delivery
from obstruction import solve_obstruction, fragility
from run_sensitivity import sweep
from run_auc_battery import fisher_auc
from run_bottleneck import forced_trajectory
from stats_utils import homogeneity_equivalence

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=257, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=64,
                  rms_norm_eps=1e-5, rope_theta=10000.0,
                  attn_implementation="eager")
model = LlamaForCausalLM(cfg).float().eval()
W = Weights(model)


class DummyTok:
    def __call__(self, s, return_tensors=None, add_special_tokens=True):
        ids = torch.tensor([(ord(c) % 250) + 1 for c in s[:16]])
        class R: pass
        r = R()
        r.input_ids = ids.view(1, -1) if return_tensors else None
        if return_tensors:
            return r
        return type("E", (), {"input_ids": ids.tolist()})()

    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


tok = DummyTok()

# --- sensitivity sweep (2x2 grid, 4 records) --------------------------------
records = []
for i, verdict in enumerate(["correct", "correct", "presence", "transport"]):
    records.append(dict(prompt=f"test prompt number {i} padding", verdict=verdict,
                        target_first_token=5 + i, source_span=[2, 3]))
res = sweep(model, tok, W, records, "cpu",
            pin_fracs=(0.25, 0.5), quantiles=(0.10, 0.25))
assert len(res) == 4 and all(0 <= r["agree_with_ref"] <= 1 for r in res)
print(f"sensitivity sweep OK ({len(res)} configs, "
      f"min agree {min(r['agree_with_ref'] for r in res):.2f})")

# --- fisher AUC --------------------------------------------------------------
rows = ([dict(verdict="presence", a=1.0 + 0.1 * i, b=0.5) for i in range(8)] +
        [dict(verdict="transport", a=2.0 + 0.1 * i, b=1.5) for i in range(8)])
a = fisher_auc(rows, ["a", "b"], "presence", {"transport"})
assert a > 0.9, f"separable classes should give high AUC, got {a}"
print(f"fisher AUC OK ({a:.3f} on separable synthetic classes)")

# --- bottleneck --------------------------------------------------------------
t = forced_trajectory(model, tok, "some prompt", "target answer", "cpu")
assert t["div_step"] >= -1 and t["n_target_tokens"] > 0
print(f"forced trajectory OK (div={t['div_step']}, "
      f"rails={t['back_on_rails']}, n={t['n_target_tokens']})")

# --- fragility ---------------------------------------------------------------
ids = torch.randint(0, 257, (10,))
C = build_cache(model, W, ids)
certify_frozen(W, C)
sig = fragility(W, C, [2, 3], power_iters=3, cg_iters=30)
assert sig >= 0 and sig == sig
print(f"fragility OK (sigma_min ~ {sig:.4f})")

# --- equivalence bound -------------------------------------------------------
labs = {"m1": ["presence"] * 40 + ["transport"] * 10 + ["selection"] * 5,
        "m2": ["presence"] * 38 + ["transport"] * 12 + ["selection"] * 5,
        "m3": ["presence"] * 42 + ["transport"] * 9 + ["selection"] * 4}
obs, bound, _ = homogeneity_equivalence(labs, n_boot=500)
assert 0 <= obs <= bound <= 1
print(f"equivalence bound OK (observed max TV {obs:.3f}, "
      f"95% bound {bound:.3f})")

print("\nALL EXTENDED SMOKE TESTS PASSED")
