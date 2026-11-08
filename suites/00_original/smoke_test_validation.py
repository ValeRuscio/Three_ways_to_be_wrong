"""Smoke test for the Sec. 4.5 validation suite: tiny Llama + tiny Pythia
through the arch-generic API, ordered labels, readout interventions, stats."""
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, GPTNeoXConfig,
                          GPTNeoXForCausalLM)

from validation_suite import (make_api, calibrate, label_failure,
                              readable_over_span, success_medians,
                              readout_interventions, Thresholds)
from stats_paper import (wilson, cochran_armitage, holm, two_prop_p,
                         logistic_fixed_effects)

torch.manual_seed(0)


def tiny_llama():
    cfg = LlamaConfig(vocab_size=257, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64,
                      rms_norm_eps=1e-5, attn_implementation="eager")
    return LlamaForCausalLM(cfg).float().eval()


def tiny_neox():
    cfg = GPTNeoXConfig(vocab_size=257, hidden_size=64, intermediate_size=256,
                        num_hidden_layers=3, num_attention_heads=4,
                        max_position_embeddings=64, rotary_pct=0.25,
                        use_parallel_residual=True,
                        attn_implementation="eager")
    return GPTNeoXForCausalLM(cfg).float().eval()


for name, model in (("llama", tiny_llama()), ("neox", tiny_neox())):
    W, api = make_api(model)
    # fabricate a mini cohort of 8 prompts
    recs = []
    caches = {}
    for i in range(8):
        ids = torch.randint(0, 257, (12,))
        C = api.build(ids)
        api.certify(C)
        caches[i] = C
        top = int(C.logits[-1].argmax())
        g = top if i < 4 else int(C.logits[-1].argsort()[-5])
        recs.append(dict(idx=i, prompt=None, target_first_token=g,
                         competitor_token=(-1 if g == top else top),
                         source_span=[3, 4], correct=g == top))
    succ = [r for r in recs if r["correct"]]
    fail = [r for r in recs if not r["correct"]]
    th = calibrate(api, caches, succ)
    med = success_medians(api, caches, succ)
    assert th.tau_q10 == th.tau_q10 and med.proj_g == med.proj_g
    labs = []
    for r in fail:
        decoys = [x["target_first_token"] for x in recs
                  if x["target_first_token"] != r["target_first_token"]][:5]
        lab = label_failure(api, caches[r["idx"]], r, decoys, th)
        assert lab in ("source", "transport", "selection")
        out = readout_interventions(api, caches[r["idx"]],
                                    r["target_first_token"],
                                    r["competitor_token"], med, seed=r["idx"])
        assert set(out) == {"answer_up", "alt_down", "combined", "random",
                            "base"}
        labs.append((lab, out))
    n_up = sum(o["answer_up"] for _, o in labs)
    print(f"{name}: labels {[l for l, _ in labs]}, answer_up recovers "
          f"{n_up}/{len(labs)}, random {sum(o['random'] for _, o in labs)}"
          f"/{len(labs)}")

# ---- stats unit checks ------------------------------------------------------
lo, hi = wilson(8, 10)
assert 0.44 < lo < 0.55 and 0.94 < hi < 0.99
z, p = cochran_armitage([40, 30, 20], [50, 50, 50])
assert z < -3 and p < 0.001          # declining trend -> negative z
z2, p2 = cochran_armitage([20, 30, 40], [50, 50, 50])
assert z2 > 3                        # rising trend -> positive z
hp = holm({"a": 0.01, "b": 0.04, "c": 0.03})
assert hp["a"] == 0.03 and hp["b"] >= hp["c"]
assert two_prop_p(40, 50, 20, 50) < 0.001
# balanced across models so the fixed effects do not absorb the trend
y, stage, mid = [], [], []
for m in ("m1", "m2"):
    y += [1] * 15 + [0] * 5 + [1] * 8 + [0] * 7 + [1] * 3 + [0] * 12
    stage += [0] * 20 + [1] * 15 + [2] * 15
    mid += [m] * 50
lf = logistic_fixed_effects(y, stage, mid)
assert lf["odds_ratio"] < 1 and lf["p"] < 0.01
print(f"stats OK: CA z={z:.2f}, logistic OR={lf['odds_ratio']:.2f} "
      f"(p={lf['p']:.1e})")

print("\nALL VALIDATION-SUITE SMOKE TESTS PASSED")
