THREE WAYS: PRIORITIZED OVERNIGHT EXPERIMENTS

Open Three_Ways_Overnight_Experiments.ipynb. It embeds this whole package.
Uses your existing Python/CUDA/Transformers 4.48.3 environment and existing HF
access. No installation or token setup. PREVIOUS=None is supported. A previous
folder must be extracted, not a ZIP. No old 7 GB upload is required for GPU jobs.

ORDER / DEFAULTS
1. Llama3.2-1B trained margin baseline, unit random readout, random-init null.
2. Llama1B patterns-only, gates-only, normalization-only frozen derivatives.
3. Qwen/Qwen2.5-3B split, then allenai/OLMo-2-1124-7B split.
4. Optional Llama1B single-layer gate-freeze matrix, last, 100 prompts.
A CPU cache analysis launches concurrently, when PREVIOUS is available.

Every campaign has a timestamp+random suffix; every job has its own new run
name/output directory. No old run is resumed or modified. The exact same cohort
is prepared once and copied into all GPU jobs. Repeated 1B jobs reuse the pinned
model revision, not their measurements. Prompt subsets use the same stable hash.

Full cohort defaults for nulls/freezes and Qwen/OLMo profiles; matrix 100 prompts.
This is potentially long. No under-one-hour claim is made. For a prespecified
100-prompt null/freeze comparison set NULL_PROMPTS=FREEZE_PROMPTS=100 before
launch. Results then describe that subset. Fixed seed is recorded. No outcome
based prompt selection. The matrix includes all seven projection families.

METHOD
- Null readout: one fixed Gaussian unit vector per seed (seed 0 default), shared
  across prompts, applied after final normalization in place of u_g-u_c. Live and
  frozen gradients use the same vector. Unit normalization changes gradient
  scale: compare selection fractions/cosines to trained margins, not raw norms.
- Random initialization: AutoConfig at pinned pretrained revision, then
  AutoModelForCausalLM.from_config, fixed seed 1729, FP32, eager attention.
  Trained weights are loaded ONLY for the matched trained/readout conditions.
  Null token IDs g,c are held equal to the trained counterpart. Null output head
  is randomly initialized with the rest of the model. No generation-based
  relabeling of null cases.
- Freezes: exact realized attention probabilities, SiLU outputs, and inverse
  normalization scales. All token positions/layers, including final norm and
  OLMo Q/K RMSNorms. Five backwards/prompt (live, all, 3 singleton).
  For each condition, content_norm is norm of the conditional gradient;
  selection_norm is norm(live - conditional). Only 'all' is the original complete
  content/selection split. Singleton effects overlap; NOT additive attribution.
  The older eight-coalition Shapley implementation remains in the source but is
  not scheduled by this notebook. Native forward equality is checked.
- Matrix: L+2 backwards/prompt (live, all frozen, L one-layer gate freezes).
  Compact scalar summaries only are written. No full gradient matrices go to
  disk. Source-layer by frozen-gate-layer effects overlap, not additive causal
  stages. Freeze j<=source i is masked in downstream figures.
- Model families/QKV bias association is observational and confounded by
  training, normalization and other architecture differences. This experiment
  does not isolate a causal effect of QKV biases.

CPU CACHE REALITY
Old channel split/geometry runs saved L x 2 x hidden subject/template origin
INPUT vectors. They did NOT save MLP gates, attention rows or first-token
pre-activation vectors. CPU output distinguishes:
  origin_input_anova.csv: exact balanced subject x template ANOVA of available
    origin input vectors, templates0/1/2, separately within relation and origin.
    One fact per subject/relation chosen before checking completeness. Missing
    triplets excluded and counted. Includes interaction residual, not a
    stochastic random-effects variance-component estimate. Cannot support a
    claim about gate/attention variance.
  first_norm_variation_PROXY.csv: variation of saved scalar first-token norms.
    NOT vector constancy; a rotating vector can have constant norm.
  sink_vector_constancy.csv: emitted ONLY with newer sink_state/*.npz caches.
    sqrt(E||v-Ev||^2)/||Ev|| per relation/template/layer/origin (population moments).
    Zero means produce undefined ratios. Near-zero means can give huge ratios.
    Nonzero-vector counts identify empty first-token origins (e.g. Qwen token0
    is subject). Mean pairwise cosine and variance fraction also retained.
  cached_gate_attention_anova.csv: available only from newer captured variance
    summaries. Attention features are semantic role masses, not raw token rows.
No cache supplied: CPU status explicitly skipped. GPU queue still runs.
If you need full gate ANOVA or sink constancy and no suitable cache exists,
use the documented separate recapture cell; it is OFF by default because it
requires GPU time. It uses three templates across Llama1B/3B/8B and Qwen7B.

OPERATIONS
Queue worker is detached. Closing browser does not stop it; powering off or
terminating the GPU instance DOES. Cooperative STOP is checked between prompts
and matrix layers. A freeze/null operation can finish before stop is observed.
Queue fails fast on GPU error. CPU errors are recorded separately. No automatic
restart/relaunch or silent resumption. Logs/exit JSON are per job and campaign.
Disk checked before EVERY GPU job including matrix. Default floor 20 GiB; this
is a user-adjustable free-space floor, not a model-download size guarantee.
Models/HF cache may need additional tens of GiB. We do not delete anything.

Small export includes summary tables/PDFs, cohort, configs, provenance, exit
status, architecture and model commits; excludes raw prompt files/states.
This is a review archive, NOT a full resume backup. Download it before deleting
the GPU instance. Keep raw data separately for arbitrary reanalysis.

TESTS
python -m unittest discover -s tests -v
CPU tiny Llama/Qwen/OLMo tests cover native/frozen gradient agreement, singleton
endpoints, full Shapley closure, matrix causality, origin certificates, finite
differences, cached ANOVA, sink statistics, report output and original workflow.
No pretrained GPU run or wall-time benchmark performed in development.
