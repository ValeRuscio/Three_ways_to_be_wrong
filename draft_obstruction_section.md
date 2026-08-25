# Draft: results section for the obstruction instrument (reframed)

Drop-in skeleton in the paper's voice. Numbers in [brackets] are from the
stand-in-label runs and should be replaced (or confirmed) once verdict-tier
labels are in; numbers without brackets are stable across both labelings by
construction (certificates, controls, within-instrument quantities).
Notation follows the paper: ob(s) is the pinned extension obstruction,
c+ the success-calibrated delivery level, S the source span.

---

## §X  The obstruction instrument

### X.1  Failure as an extension obstruction

The sheaf formalism of Section 3 makes a promise the ledger alone does not
test: that transport failure is a *local-to-global extension* event. We now
make that promise operational. Fix the realized activations on the causal
cone of the source span, pin the source section through layer L/2 and the
off-cone context everywhere, and ask for the minimal frozen-dynamics defect
energy required to raise the terminal delivery ⟨u_g, x̂_{L,T}⟩ to the
success-calibrated level c+:

    ob(s)² = min_δ  Σ_ℓ ‖X_{ℓ+1} − F_ℓ(X_ℓ)‖²   s.t.  pins, delivery = c+.

Because the realized trajectory satisfies every dynamics constraint exactly
(the reconstruction certificates of Table 1), ob(s) vanishes if and only if
the pinned extension exists: the section over S extends to a
target-delivering terminal state within the selected sheaf. The demand is
one-sided — a delivery surplus is not an obstruction — and the residual is
computed matrix-free by conjugate gradients, with the adjoint exact because
the frozen map is linear (certificate C3: the residual operator is affine to
[3×10⁻⁴] relative in fp32, [10⁻⁸] typical).

Under this instrument, the trichotomy becomes three statements about one
scalar and one sign. A **selection failure** is an example with ob(s) = 0,
delivery surplus positive, and terminal margin negative: the section
extends, delivery clears the success level, and the answer still loses the
terminal pairing. This is not a definition we impose; it is what the solver
returns. In every model, every failure our verdict rule labels selection has
ob(s) = 0 *exactly*, with positive surplus. A **transport or presence
failure** is an example with a delivery shortfall and ob(s) > 0; the two are
separated not by ob but by the source content π_S that the extension has to
work with.

### X.2  Separation and ordering

Across the seven primary models the class medians are strictly ordered,
presence > transport > selection = correct = 0, in [6/7] models
([per-model table; e.g. Llama-3.1-8B: 0.57 / 0.34 / 0.00 / 0.00]). The AUC
for separating {presence, transport} from {selection, correct} by ob alone
is [0.94–0.99] per model. Raw ob is a within-model quantity — logit scales
differ across families (Qwen2.5 medians run 5–10× the Llama values) — so
cross-model comparisons use ranks or success-normalized units.
[The one ordering violation is Qwen2.5-7B (transport median above
presence); this is the same checkpoint that produced the transported-support
ablation anomaly of Section 6.9, and we flag the coincidence without
explaining it.]

### X.3  Discrimination without calibrated thresholds

The verdict rule of Section 4 depends on success-calibrated thresholds. The
obstruction does not: it is a single canonical scalar per example. Table [Y]
compares three feature tiers for failure-class discrimination — black-box
(confidence, entropy, top-2 margin), the paper's internal summaries (π_S,
τ), and the sheaf tier (ob, delivery surplus) — pooled across models. The
sheaf tier matches or exceeds the internal tier on every task where the
comparison is clean, and dominates black-box confidence throughout
([transport-vs-rest 0.94 vs 0.76]; [presence-vs-transport 0.94 vs 0.71]).
The one task where black-box features are competitive is selection-vs-rest
([0.75] both) — as expected, since selection failures are precisely the ones
whose target logits look healthy from outside; each tier is best at
detecting the failure class its variables define.

[Footnote for the circularity: with stand-in labels the internal tier's
presence-vs-transport AUC is exactly 1.000 because π_S participates in the
label definition; the table's headline comparisons therefore use
verdict-tier labels, under which no feature tier defines the classes.]

### X.4  Stability

The two calibration choices the instrument does retain — the depth to which
the source section is pinned, and the success quantile defining c+ — are
swept in Table [Z]. Pin depth is irrelevant everywhere (agreement 1.00
across L/4, L/2, 3L/4 at every quantile). The quantile matters only when
pushed below the success distribution's own tail: at Q0.05 two models'
transport medians collapse to zero and the AUC degrades ([0.71–0.95]),
while the operating range Q ∈ {0.10, 0.25} is uniformly strong
([AUC 0.95–0.99]). We report this as the shape of the instrument's validity
region rather than as uniform insensitivity.

### X.5  Task families

The extraction prediction of Section 3.5 is reproduced by the instrument
with no access to the verdict rule: across all seven models, synthetic
key–value extraction produces **zero presence failures** — the target is in
the context by token identity, and every failure is a delivery shortfall
(transport, [0.42–0.90] of failures) or a lost terminal competition
(selection, the remainder). Two-hop composition over the PopQA join behaves
accordingly at hop level [numbers from the popqa-join cohort].

### X.6  Interventions as validation probes

The interventions in this section are not proposed as fixes. They are
validation probes: if the diagnostic says a failure involves source
evidence, transported support, or terminal competition, then perturbing the
corresponding realized structure should move the target margin more than
matched random controls. This weaker test is the appropriate one for a
diagnostic, and it is passed with large margins: every diagnostic-guided
intervention family moves the mean margin by [0.4–0.9] logits against
random-head controls at [|Δ| < 0.05] (Table [W]).

Two results exceed the weak criterion. First, selection failures are
causally dissociated from the other classes: they are insensitive to donor
source-patching ([+0.17]) and to forced transport ([+0.10]) — consistent
with delivery already being at success level — but respond to demoter-head
ablation ([+0.62]). Together with the ob(s) = 0 signature and the
continuation geometry below, selection is characterized three ways —
observationally, causally, and geometrically — by independent instruments.
Second, the two transport arms dissociate *mechanism*: re-weighting
attention probabilities on the top transported-support edges is inert
([+0.13] on transport failures), while forcing the same heads to carry the
source value message is not ([+0.47]). Transport failures respond to what
an edge delivers, not to its attention weight — an interventional
counterpart to the observation of Section 2 that attention mass is not
transported support.

We do not claim class-specific repair. The presence and transport arms
remain entangled at this intervention granularity: donor source states
patched at mid-depth also help transport failures ([+0.63]), plausibly
because mid-depth donor states already contain partially transported
support, and forcing transport moderately helps failures labeled presence
([+0.40]), plausibly because presence is a graded quantity and weak source
sections still carry something. Distinguishing label noise from mechanism
coupling here requires the verdict-tier labels and an early-band source
patch, which we leave as specified in Appendix D. Diagnosis is not repair;
repair is an external use case of diagnosis, and the probes above test only
that diagnosed structures are causally live.

### X.7  Continuation geometry

Forced-decoding trajectories stratify by verdict in the direction the
taxonomy predicts. Among failures, the back-on-rails rate — forced past the
divergence token, the model completes the target — is [0.61] for selection,
[0.37] for transport, and [0.31] for presence (pooled; the ordering holds
per model). A selection failure sits one token from a recoverable
trajectory; a presence failure has nothing to get back onto the rails of.
Divergence position is uninformative in this task family (entity answers
diverge at the first content token by construction) and is reported only
for completeness.

### X.8  Failure mixtures are model- and instrument-dependent

Under the obstruction-based labels, the coarse failure mixture varies
substantially across the seven models (presence share [0.12–0.52]; maximum
pairwise total-variation distance [0.47], 95% bootstrap bound [0.62]).
We treat this measured variation as part of what the diagnostic is for:
the instrument reports the mixture rather than assuming one.

[Tension paragraph — must appear, in some form:] This appears to stand in
tension with Section 6.1, which does not reject homogeneity of the mixture
across models under the verdict-tier labels. The two results are not
directly comparable: the cohorts differ, and the labelers differ — the
verdict tier's presence test is decoy-controlled readability, while the
obstruction labels presence by source content against a success quantile.
[Resolution to run: apply both labelers to one cohort; report the confusion
matrix and both mixtures. If homogeneity holds under the verdict tier and
not under the obstruction, the discrepancy localizes to the presence
boundary, and Section 6.1's claim should be stated as instrument-relative.
If it fails under both, Section 6.1 should be weakened accordingly.]

### X.9  [Optional] Controlled ontogeny

[If the toy results hold across seeds:] On a synthetic corpus in which
exposure counts are assigned rather than observed, the diagnostic tracks
training as the theory predicts: presence failures are the entire mixture
at initialization, transport becomes a stable class as accuracy rises, and
selection appears last; at convergence, residual failures concentrate in
the lowest exposure band (presence) and among facts whose object is shared
by many subjects (selection). Because exposure is assigned, this is the
interventional form of the long-tail association of Section 6.1.

---

## Additions to Limitations

- The obstruction's residual depth profile does not localize the failure:
  defect energy concentrates near the terminal cut for every class, because
  the delivery constraint acts only there. Localization requires a
  transported-support (rather than terminal) delivery demand; we report the
  centroid's failure as a negative result. [This also explains the
  pin-depth insensitivity in X.4.]
- The demoter-ablation probe selects heads per example by their realized
  anti-margin contribution; it validates that terminal competition is
  causally live but is not verdict-specific evidence by itself. A held-out
  (family-level) demoter set is the sharper probe.
- All obstruction-label results in this section use the ordered
  delivery/content rule calibrated per model; [statement of which tables
  were re-derived under verdict-tier labels].

## One-sentence additions elsewhere

- Abstract (optional): "A certified extension-obstruction instrument
  derives the trichotomy from a single scalar: selection failures are
  exactly the wrong answers whose pinned extension exists."
- Discussion: "Diagnosis is not repair; repair is an external use case of
  diagnosis. The interventions above test only that what the diagnostic
  points at is causally live, and the toy-ontogeny experiment illustrates
  the diagnostic's value where no repair question arises at all."
