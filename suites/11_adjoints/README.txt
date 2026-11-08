THREE WAYS: ACTIVATION ADJOINTS AND RESTRICTED TRAINING UPDATES

Run Three_Ways_Adjoints_and_Learning.ipynb on your existing Lambda environment.
No installation or authentication cells. Tested dependency: Transformers 4.48.3,
FP32 PyTorch, eager attention. Bundled cohort and model pins from the prior run.
No previous multi-GB results required. Model downloads may still be needed.

DEFAULT: experiments 1+2, 100 matched prompts each on Llama-1B and Llama-8B.
Two backward passes per prompt, one live and one fully frozen, plus native capture
and a zero-embedding frozen replay certificate. All token positions, all residual
cuts, role-level metrics, input-origin margin contributions. Final cut is before
final normalization: late agreement is not guaranteed because that norm remains
input-dependent in the live derivative. No full-vocabulary Jacobians stored.

OPTIONAL flags in notebook configuration, enabled BEFORE first launch:
RUN_JACOBIAN: experiment 3, 100 prompts on 1B, eight VJPs per layer. Rebuilds one
forward graph per layer to bound memory. Rademacher probes estimate squared
Frobenius norms; saves probe variance and raw samples. Total and per-position norms
reported because subject/template token counts differ. Save first crossing and
three-consecutive-layer crossing; no crossing stays missing. Relation-specific
regime-success contrasts report crossing rates separately from conditional depth.
Do not call this the same estimand as the population template-by-subject ANOVA.

RUN_SGD: experiment 4, 100 prompts on 1B, gate/down matrices at first/middle/last
layers, each independently, at three norm-relative step sizes. Update direction
is negative first canonical gold-token CE gradient; effective learning rate saved.
Each edit is restored exactly, with no accumulated training. Gold/competitor fixed
from baseline; top-1 checked over full vocabulary. No full-answer repair claim.
Content prediction is exact ONLY for one chosen matrix with all mechanisms frozen.
Measures native finite change, actual frozen finite change, content first-order,
selection first-order, and native nonlinear remainder. For gate/Q/K weights the
fully frozen content derivative vanishes by construction: not evidence that a fact
was absent. No random-weight control or specificity test is claimed by this suite.

RUN_CURVATURE: experiment 5, 8 prompts on 1B. One live and one frozen HVP per
selected matrix, baseline margin-gradient direction. Also checks centered second
differences. Frozen within-matrix Hessian zero; simultaneous-matrix cross-curvature
can be nonzero even with mechanisms frozen. Numerical differences can be unresolved.
Defaults gate/down, first/middle/last layers; editable weight_families/layer fractions.

Origin groups: subject, first, template are disjoint, subject takes priority if
first token overlaps. Answer site = last prompt position; it is an overlapping
view, usually in template. subject_last is another overlapping view. Concatenated
role gradients use Frobenius geometry, not an average of per-token cosines.
||selection||/(||selection||+||content||) is descriptive, NOT additive energy or a
percentage of causal influence. Signed cross term and closure are included.

REGIMES: Bundled independent-calibration origin labels (quantile .05), not validated
causal stages; include answer_success, continuation/uncalibrated/mixed failures.
Sparse support is retained, not silently forced to absent/overridden/misretrieved.
Reference source and definition are in evidence/regime_provenance.json. Optional
custom regimes_csv requires model,id,template,regime and consistent duplicate keys.
Do not transfer labels from another model revision, prompt or diagnostic definition.

STATISTICS: equal-fact mean within each reported group, subject-cluster bootstrap.
Pointwise confidence intervals, conditional on labels/model/reference cohort; no
multiplicity correction or calibration uncertainty. No CI with <5 subjects.
Depth conditional on crossing is selected; inspect missing rates and probe SE.
Per-template/per-relation raw measurements retained for reanalysis.

OPERATIONS: detached supervisor + worker locks; atomic per-prompt JSON; identical
protocol resume only. STOP file requests stop between prompts or optional-stage
layers/matrices. Keep GPU instance running; browser can close. Changed config means
new RUN_NAME. No huge caches saved. Report-only snapshots read completed records.
Export ZIP includes scalar per-prompt data, CSV/PDF/PNG reports, source/protocol,
cohort/model pins/regime labels and bounded log tails; no downloaded model weights.

CLI (from extracted source, using existing Python):
python -m adjoint_study.supervisor --config /absolute/configuration.json --out /absolute/new_results --evidence ./evidence
python -m adjoint_study.run --config /absolute/configuration.json --out /absolute/results --evidence ./evidence --report-only

VALIDATION: tests/test_study.py runs tiny random Llama/Qwen2/OLMo2 models to verify
frozen replay adjoints, embedding finite differences, exact Jacobian vs Monte Carlo,
SGD descent/restoration/linearity, zero frozen Hessian, reporting and resumability.
These are implementation tests, not pretrained-GPU speed or scientific benchmarks.
