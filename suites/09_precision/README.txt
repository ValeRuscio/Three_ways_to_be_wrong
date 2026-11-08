THREE WAYS: CERTIFICATES, RECALIBRATION, 8B FREEZES AND ANISOTROPY

Open Three_Ways_Precision_Experiments.ipynb. All source, your saved cohort,
model revision pins and compact input tables are embedded. No previous 7 GB
upload, environment installation, kernel setup or HF token cell is needed.
Use the same existing Transformers 4.48.3 / PyTorch CUDA environment.

DEFAULT GPU QUEUE
- Gradient certificates: all six previously measured models; 16 fixed prompts
  per model; three depths (first, middle, last); epsilon 1e-5,1e-4,1e-3.
  Seven projection families; one random rank-one direction and one leading
  live-gradient rank-one direction per matrix where nonzero.
- Llama3.1-8B singleton freezes: all 2,995 evaluation prompts; live, all frozen,
  attention only, gates only, scales only (five backward passes per prompt).
- Anisotropy: Llama3.2-1B, 100 fixed evaluation prompts, 100 independent reference
  subjects; trained model and three initialized models (seeds1729,1730,1731).
- CPU report-only calibration runs concurrently.
Instruct pairs are omitted. No family-level observation becomes an instruct or
causal QKV-bias claim. Defaults are explicit and editable before launch.

COST
Not benchmarked on GH200; no 'minutes' guarantee. Certificates have up to
3 depths x 7 families x 2 directions x 3 epsilons =126 checks per prompt.
Each check uses two live evaluations and one frozen replay, hence up to378
forward/replay evaluations per prompt, plus gradient/capture/generation work.
At16 prompts this is6,048 evaluations per model.
Full 8B freezes may take hours. Select100 fixed prompts explicitly for a smaller
study; this is not silently done. Anisotropy uses four backward passes/prompt:
residual live/frozen plus weight-factor live/frozen, with independent forward
reference captures. Completed per-prompt outputs are saved as they finish.

CERTIFICATES
- Live central finite difference (m(W+aUV)-m(W-aUV)) versus2a<G_live,UV>.
- Frozen finite response versus a<G_content,UV>, holding the base program fixed.
- Delta Frobenius norm a=epsilon*||W||F, normalized rank-one directions.
- Native frozen reconstruction tolerance1e-4; FP32, eager attention, TF32off.
- Checks use absolute + relative tolerances; predicted changes below10*atol are
  unresolved. Frozen gate/Q/K weight derivatives are structurally zero and
  must be reported as such, not evidence of resolving a nonzero slope.
- Reports contain EVERY check, failures, maxima, unresolved counts, actual and
  predicted changes, epsilon, direction, and live/content/selection norm identity.
- Do not change tolerances after inspecting errors to manufacture a pass.
  Inspect epsilon trends: truncation dominates large steps and rounding small.
- These are sampled numerical certificates, not a proof for every weight or
  every quantity. Raw failures are retained; queue completion does not mean
  every numerical check passed. Inspect certificate_audit.json.

REPORT-ONLY CALIBRATION
Input tables are from your completed reviewer_v1 archive. Nothing is captured.
Original independent calibration labels are preserved. A restricted8B/OLMo
view retains unresolved cases; neither model has perfect original coverage.
Exploratory alternative: five-fold subject-held-out calibration using existing
successful canonical-answer evaluation prompts across all five templates.
Every subject is wholly excluded from the fold's calibration, across templates
and relations. Each calibration subject contributes one mean. References are
relation-specific; no global fallback; failure cases do not fit thresholds.
Quantiles .01,.05,.10 and minimum successful subjects5,20 are both reported.
Min5 improves coverage but remains sparse. Min20 may REDUCE coverage. Neither
creates new independent successes. Do not rename this scheme the old labels.
Default primary claims should retain original independently calibrated labels;
use crossfit results as an exploratory sensitivity analysis.

The existing tables were tested locally: at q=.05, min5 crossfit leaves7.3%
of1B errors uncalibrated vs75.0% originally, but min20 leaves83.9%. This is why
'cost-free recalibration fixes the evidence' is not an honest conclusion.
No CIs for label prevalence are manufactured: proper uncertainty would refit
calibration under subject resampling. Counts are descriptive.
Original per-regime aggregate gradient summaries cannot be relabelled. If you
supply raw profile directories, the optional join rebuilds profile summaries
from per-prompt split.json after checking exact model SHA. Otherwise that join
is explicitly skipped; all label/coverage tables still work with bundled data.

ANISOTROPY: DEFINITIONS AND REQUIRED NEW CAPTURES
Previous null outputs contain scalar statistics, not gradient factors or
residual means. A CPU-only calculation is impossible from those summaries.
A weight-matrix gradient and a residual vector have different shapes. We report
TWO explicitly compatible quantities, not an ambiguous signed 'weight cosine':
1) Signed cosine between d margin/d x_l and an independent mean residual mu_l,
   where x_l is the block-entry residual vector at the final prompt token.
   Compute live, frozen-content, and difference gradients; also compare with
   each prompt's own x_l. These are RESIDUAL-SPACE derivatives, not d margin/dW.
2) For down and attention-output matrices only (their outputs are residual
   coordinates), ||mu_hat_l^T G||_2 / ||G||F. This is the nonnegative Frobenius
   subspace-projection cosine onto matrices with output direction along mu_l,
   not a signed vector cosine. G is the exact full-token weight derivative.
Means use100 subject-disjoint calibration prompts (available template0), with
no correctness filtering; are estimated separately within each initialized or
trained model. Norms/mean-energy fractions are saved. Evaluation objectives
use fixed gold and trained competitor tokens for all initialized counterparts.
Layer/seed results remain explicit, not one arbitrary pooled number. Confidence
intervals bootstrap evaluation subjects, conditional on reference sets/seeds.
Reference-set uncertainty is not included. Three seeds are still a limited
initialization sample. This geometry is diagnostic, not proof of a causal
normalization-compensation mechanism. Native RMSNorm effects and residual
anisotropy can coexist without one alone explaining the other.

OPERATIONS
Fresh campaign and per-job folders; all cohort/model pins preserved. Configuration
cell refuses to overwrite the notebook's selected campaign, to prevent exporting
an accidentally empty new folder. Detached queue, incremental files, cooperative
STOP, free-disk check before each job, bounded log tails and explicit exit records.
No automatic resumption. Closing browser is fine; shutting down instance stops
workers. CPU jobs may finish after the GPU portion; monitor shows both.
Small export includes summaries and per-check numerical certificate tables,
cohort/config/provenance, and log tails; excludes model activations/raw cache.
It is for review, not a full resume backup. Download before deleting GPU instance.

SOURCE CLI
python -m precision_study.recalibrate --evidence evidence --out cpu_results
python -m precision_study.queue --plan /absolute/path/to/plan.json
python -m unittest discover -s tests -v
