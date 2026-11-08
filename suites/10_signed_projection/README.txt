SIGNED RESIDUAL-MEAN PROJECTION UNDER SINGLETON FREEZES

Open Three_Ways_Signed_Projection.ipynb. All code, the existing cohort and exact
Llama1B/8B model revision pins are embedded. No previous 7GB upload, install,
venv, kernel or HF login setup is needed. Uses tested Transformers4.48.3,
PyTorch/CUDA FP32 eager attention, existing model access.

QUESTION AND DEFINITIONS
At block entry, at the final prompt token, let G_S be the derivative of the
fixed gold-minus-competitor logit margin with respect to the residual vector
when mechanisms S are held at their native-pass values. Let u_l=mu_l/||mu_l||,
where mu_l is an independent reference mean at that same site.
Measure p_S=<G_S,u_l> for:
  live (no freezes), attention-only, gates-only, scales-only, all-frozen.
Patterns=attention probabilities; gates=SiLU outputs; scales=inverse RMSNorm
scales, including final norm. Whole-network freezes; no local-layer attribution.
Native forward prediction is preserved by every freeze. The residual gradient
at layer l can only depend on freezes in its downstream computation.

PRIMARY EFFECT: E_S = p_live - p_S.
Positive E_S means permitting that mechanism adds a positive mean-direction
component in this conditional derivative comparison. If p_all is negative,
p_live near zero, p_scales moves back negative, and E_scales is more positive
than the other E values, the results support a scale-mediated compensating
component. The code reports signed quantities and paired contrasts, not an
automatic 'mechanism proved' threshold or a percentage of error explained.

Report both raw projection and shared-denominator normalized projection,
p_S / ||G_all||, and E_S / ||G_all||. Each prompt uses the SAME denominator
for all five conditions. Per-condition cosine is secondary: cosine changes
both angle and its denominator, so a cosine difference is not an additive
projected effect. Near-zero mean/gradient denominators return undefined values.

Paired comparisons are E_scales-E_gates and E_scales-E_attention. An additional
singleton-sum-minus-total column measures nonadditivity. Singleton effects
can overlap/cancel; do not sum them as independent shares. This is a derivative
intervention, not a finite weight edit or proof that a factual error is repaired.
Signed residual-space results are not signed weight-matrix cosines.

MATCHED DESIGN
Defaults: trained Llama3.2-1B and Llama3.1-8B;100 evaluation prompts each,
100 independent calibration/reference subjects, fixed seed0. Exact same
selection code as the previous anisotropy run (evaluation hash seed7 and
reference hash seed0). It selects100 prompts, not100 distinct subjects: the
current cohort gives90 evaluation subjects. One template0 reference prompt
per calibration subject, no correctness filtering. Entire reference subjects
excluded from evaluation. Each model gets its own reference mean, but that
mean stays identical across all five freezes. This is a new targeted run,
not a repeat of all earlier experiments. Initialization controls are not rerun.

CONFIGURATION
models=['llama1b','llama8b']; edit before first launch to use1B only.
prompts=100; set None for full2,995 evaluation prompts if intentionally wanted.
reference_subjects=100; keep unchanged for matching previous means.
save_gradients=True: retains five compact residual-gradient vectors per layer
per prompt. Roughly313MiB uncompressed for both default models, excluding model
weights. Reference means and scalar measurements are saved atomically.
Gradient caches are excluded from the small review ZIP but remain on GPU disk.
The existing Llama8B FP32 weights alone need roughly32GB of accelerator memory.
No new runtime benchmark was performed; count is five backwards/prompt/model,
plus reference captures and small finite-difference checks.

VALIDATION
Native-versus-frozen baseline objective tolerance=1e-4+1e-5*abs(native margin).
Native affine reconstruction tolerance1e-4.
Two prespecified evaluation prompts/model receive finite differences along u_l
at three depths and epsilons1e-5,1e-4,1e-3. Delta norm=epsilon*||x_l||.
For EACH of five conditions, inject the perturbation into the final-token
block-entry residual and compare centered response with2*delta*p_S.
Freezes remain at baseline values. Larger steps can fail in partially live
conditions due to nonlinearity. Tiny signals flagged unresolved separately.
Reports retain failures, unresolved counts, predicted/measured differences.
A successful worker exit is not a claim all finite checks passed.

STATISTICS AND OUTPUTS
projections_per_prompt.csv: full scalar measurements for all five conditions.
projections_summary.csv: equal-fact means and subject-cluster95% bootstrap CIs.
paired_contrasts_summary.csv: compute paired contrasts BEFORE bootstrapping.
by_template_summary.csv: exposes variation across templates/reference choice.
coverage_and_baselines.csv: IDs, tokens, margins and reconstruction checks.
finite_difference_per_check.csv and finite_difference_summary.csv.
audit.json and model-specific PDF/PNG figures.
Confidence intervals condition on model weights and the100 reference subjects.
They do not include reference-set or model-training uncertainty, and are not
multiple-testing-adjusted. No error-regime labels are inferred in this study.

OPERATIONS
Use the notebook configuration cell ONCE. It refuses to change RESULTS on a
repeat, avoiding accidental export of a new empty run. Launch is detached;
closing browser is fine, terminating the GPU instance is not. Monitor the actual
worker log and exit.json. STOP is cooperative at reference/evaluation boundaries.
Every completed prompt is saved. Explicit resume requires same protocol/source/
model pins/cohort, and skips completed records. Never start a second GPU worker.
Small export checks completion and includes scalar reports, means, config,
provenance and log tails; download it before terminating the instance.

CLI
python -m projection_study.supervisor --config CONFIG --out RESULTS --evidence evidence
python -m projection_study.run --config CONFIG --out RESULTS --evidence evidence --report-only
python -m unittest discover -s tests -v
