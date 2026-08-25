# Paper mapping ("Three Ways to Be Wrong", TMLR submission)

| Paper artifact | Script / notebook | Output |
|---|---|---|
| Table 1 (natural-task categories) | `run_paper.ipynb` (obstruction stage) | `results/<tag>/obstruction.csv` |
| Table 2 + Fig. 3 (Sec. 4.5 validation suite) | `run_validation.ipynb` / `run_validation_suite.py` | `results/<tag>/validation_suite.csv` |
| Table 3 (capability matrix) | documented per-arch: `frozen_cache.py` (Llama/Qwen/OLMo), `frozen_neox.py` (Pythia) | certificates in every run |
| Table 4 (reconstruction errors), Table 5 (runtimes) | `run_profile.py` | `results/<tag>/profile.csv` |
| Table 6 (label AUCs) | `run_auc_battery.py` | `results/<tag>/auc_battery.csv` |
| Table 7 (constrained delivery, App. D) | `run_ob_stress.py` | `results/<tag>/ob_stress.csv` |
| Tables 8-9 (removal prediction, App. H) | `run_ranking.py` (post-audit gradxact sign) | `results/<tag>/ranking.csv` |
| Fig. 2 A/B (controlled circuits) | `known_circuit2.py` (12 seeds, 4 conditions) | `results/known_circuit2/` |
| Fig. 2 C (target substitution) | `run_ranking.py --specificity` | `results/<tag>/specificity.csv` |
| Sec. 4.4 (first-token / rails by label) | `run_bottleneck.py` | `results/<tag>/bottleneck.csv` |
| Fig. 4 + App. C (exposure) | `run_exposure.py` | `results/<tag>/exposure.csv` |
| App. F (prompt variants) | `run_reliability2.py` | `results/<tag>/reliability2.csv` |
| gradxact sign convention | `audit_gradxact.py` (analytic calibration; run before ranking) | printed verdict |

Smoke tests (run all on any new environment): `smoke_test.py`,
`smoke_test_extended.py`, `smoke_test_tool.py`, `smoke_test_validation.py`.

# Delivery-residual validation & verdict-specific repairs

Six experiment suites for the ledger paper:

1. **`run_obstruction_validation.py`** — computes the pinned extension
   obstruction `R(S)` and tests whether it (a) separates transport failures,
   (b) agrees with the existing transported-support verdicts, (c) predicts
   ablation effect sizes better than raw attention and `tau`, (d) transfers
   across models.
2. **`run_repair_matrix.py`** — verdict-specific causal repairs with
   cross-class controls: source-state patching (source), tDLA edge boosting
   (transport), demoter-head ablation (selection), random-head control.
3. **`run_sensitivity.py`** — pin_L x c_plus-quantile sweep; shows the verdict
   decomposition is stable under the calibration choices (the
   threshold-arbitrariness defense).
4. **`run_auc_battery.py`** — black-box vs paper-internal vs affine features
   for source/transport/selection discrimination; if ob+centroid matches
   pi_S+tau, the residual is a self-contained diagnostic.
5. **`run_bottleneck.py`** — first-token bottleneck (divergence step,
   back-on-rails) resolved by verdict class; requires `target_text` in the
   cohort. Prediction: bottleneck concentrates in selection/transport.
6. **`fragility()` in `obstruction.py`** — normal-equations spectral gap
   (sigma_min of the pinned dynamics operator) via inverse power iteration.
   Prediction: small gap = fragile transport; correlate with flip radii.

`stats_utils.py` turns the cross-model homogeneity claim into an affirmative
95% equivalence bound on pairwise TV distances (run over all models'
`obstruction.csv`). Notebooks: `run_experiments.ipynb` (suites 1-2),
`run_extended.ipynb` (suites 3-6). Smoke tests: `smoke_test.py`,
`smoke_test_extended.py` — run both with your ml_env python on any new
environment or transformers version.

## Files

| file | contents |
|---|---|
| `frozen_cache.py` | coefficient/state cache, frozen linear map, exact pullback tDLA, certificates C1–C2 |
| `obstruction.py` | pinned least-squares `R(S)` via matrix-free CG, certificate C3 (affinity) |
| `repairs.py` | live-model interventions (wrapped eager attention, residual patch hooks), certificate C-live |
| `run_obstruction_validation.py` | experiment 1 driver |
| `run_repair_matrix.py` | experiment 2 driver |

## What `R(S)` is

Unknowns: residual states on the causal cone of the source span (layers
`1..L`, positions `>= min(S)`). Pins: layer 0; positions left of the cone
(satisfied by causality); the source span through layer `pin_L` (default
`L/2`) — this pinned restriction is the fixed source state. One weighted row
demands success-calibrated target delivery at the final-position state
(`c_plus` = median `<u_g, x_hat_{L,T}>` over matched successes).

The realized trajectory satisfies every dynamics row exactly, so
`R(S)` = minimal frozen-dynamics defect energy needed to deliver the target
while holding the source section and context fixed. `R(S) ~ 0` iff the pinned
extension exists (the connecting-map obstruction class vanishes); the reported
number is the norm of its least-squares representative.

**Predicted signatures (what experiment 1 tests):**

- correct & selection: small `ob` (delivery already at/near success level);
- transport: large `ob`, residual energy concentrated mid-route;
- source: large `ob`, residual energy concentrated at the source / early
  layers (the solver must *create* the signal, not move it);
- so magnitude separates `{source, transport}` from `{selection, correct}`,
  and the **depth centroid** separates source from transport.

## Cohort schema (`cohort.jsonl`)

```json
{"prompt": "The capital of Maryland is",
 "target_first_token": 12345,
 "competitor_token": 67890,
 "source_span": [4, 5],
 "verdict": "transport",
 "tau": 0.031,
 "ablation_effect": 1.42,
 "donor_prompt": "Annapolis is the capital of Maryland. The capital of Maryland is",
 "donor_source_span": [10, 11]}
```

`tau`, `ablation_effect`, `donor_*` are optional (`null` ok). `source_span`
is inclusive token indices under the *model's* tokenizer and must be
contiguous. Export these from your existing verdict-tier pipeline.

## Running

```bash
pip install torch transformers scipy
python run_obstruction_validation.py \
  --model meta-llama/Llama-3.2-3B \
  --cohort cohorts/llama32_3b_parametric.jsonl \
  --out results/llama32_3b --run_ablation
python run_repair_matrix.py \
  --model meta-llama/Llama-3.2-3B \
  --cohort cohorts/llama32_3b_parametric.jsonl \
  --out results/llama32_3b
```

fp32 + eager attention throughout (as in the paper's exactness protocol).
Cross-model table = concat of the per-model `obstruction.csv` files.

## Design choices that are science, not code (decide & report)

- `pin_L` (default `L/2`): how deep the source section is pinned. Sweep
  {L/4, L/2, 3L/4} and report verdict-share stability — this doubles as the
  threshold-sensitivity analysis the reviewer will want.
- `lam` (default 30): terminal-row weight. Check `delivered_gap` at optimum is
  ~0; if not, raise `lam`.
- `c_plus`: median success delivery. Alternative: the paper's `Q+_{0.10}`
  quantile, for consistency with the verdict rule.
- Source-repair donors: same fact under a different template or with
  supporting context (the context-dominance treatment). Donor and receiver
  source spans must be aligned in length.
- The fast ablation screen in experiment 1 runs inside the frozen account;
  if you have live-model ablation effects from Sec. 6.9, pass them via
  `ablation_effect` instead — they are the ground truth.

## Certificates

Every run asserts: C1 manual forward == model logits (<1e-4 rel, fp32);
C2 frozen stack reproduces all states & logits; C3 the obstruction residual
operator is affine (linearity of the frozen map, hence exact adjoints);
C-live wrapped attention with identity edit == unwrapped model. If C1 or
C-live fails, suspect a transformers version drift in attention/RoPE — the
manual forward in `frozen_cache.py` is the single point of truth to fix.

## Known scope limits

Llama/Qwen2.5-style decoders only (RMSNorm, full RoPE, gated SiLU MLP, GQA,
optional QKV biases). Qwen3 (QK-norm), Gemma-2 (softcapping), Pythia
(parallel blocks) are outside this contract, matching the paper's capability
matrix. Sliding-window attention not implemented (fine at these prompt
lengths). `resid_patch` assumes batch size 1.
