# Experiment index

Numbers refer to `3_ways_new.pdf` reviewed 8 November 2026. They may change in the final manuscript. Entries identify generating studies, not a claim that every publication layout is recreated pixel-for-pixel.

| Manuscript result | Source / notebook | Notes |
|---|---|---|
| Table 1, controlled source identification and removal rankings | 01_controlled_and_validation | `revision/controlled.py`, `revision/natural_ranking.py`; original studies also in 00_original |
| Target substitution; legacy removal rankings (Appendix G) | 00_original | `run_ranking.py`, `ranking.py`; historic code and README retained |
| Figure 1 / Table 2, crossed gate and attention-role variance | 07_reviewer_followups | `followup_study/features.py`, `run.py`, `report.py`; stages `variance` and `sink` |
| Figures 2 and regime Tables 3–4, 8–9 | 07_reviewer_followups + 09_precision | Independent-calibration `regimes`; CPU `precision_study/recalibrate.py` for cross-fit sensitivity |
| Readability failure plants and sensitivity (Section 7.2 / Appendices D–E) | 01_controlled_and_validation + 02_causal_interventions | Historical classifier; not validation of the new origin labels |
| Routing amplification (Section 7.3) | 02_causal_interventions | Live attention interventions; group effects do not imply per-prompt sign accuracy |
| Matched-template gate factors (Section 7.4) | 04_gate_experiments | Factor/preactivation study; inspect matched-case coverage |
| Figure 3, nulls and channel profiles | 06_channels + 08_overnight | Additional models and matched nulls in overnight queue |
| Table 5, singleton freezes on 1B and 8B | 08_overnight + 09_precision | Condition-specific norm fractions and cosines |
| Figure 4, source-layer × frozen-layer matrix | 08_overnight | Last queue stage, explicitly capped prompt count |
| Figure 5, live/frozen activation adjoints | 11_adjoints | `adjoint_study`; default stage `adjoints` |
| Figure 6, signed mean-residual projection | 10_signed_projection | `projection_study`; shared normalization denominator |
| Trained versus initialized anisotropy | 09_precision | `precision_study/anisotropy.py` |
| Table 7, gradient certificates | 09_precision + 10_signed_projection + 11_adjoints | Separate per-pass reconstruction from subsampled derivative checks |
| Appendix H, one-matrix learning steps | 11_adjoints | Enable SGD; canonical first-token loss, edits restored after each dose |
| Appendix I, curvature | 11_adjoints | Enable curvature; eight-prompt default |
| Appendix J, sparse gate/weight/donor edits | 03_donor_followups, 04_gate_experiments, 05_pattern_experiments | Negative/exploratory results; older cohorts and protocols |
| Appendix J, per-prompt Jacobian crossover | 11_adjoints | Enable Jacobian; first and persistent crossings are separate |
| Withdrawn readout-repair prediction | 01_controlled_and_validation | Legacy utility analysis; do not present as incremental validation |

## Historical and current definitions

Suite 00 is the original repository snapshot. Suite 01 contains the initial reviewer-response experiments. The origin classifier used in the later manuscript is in `07_reviewer_followups/followup_study/regimes.py` and its versioned copies in the later suites. Preserving old code is necessary for the negative results but does not endorse old interpretations.

## Results coverage

The release includes compact archived results for channels, reviewer follow-ups, completed overnight, precision, signed projection and adjoints. Earlier controlled/causal/gate/pattern scalar results are not included in this compact subset; their generating source is included. Re-running those experiments may require the input specified in their notebooks. The observed reports are not a complete raw-data release.
