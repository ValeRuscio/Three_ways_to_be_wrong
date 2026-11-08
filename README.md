# Three Ways to Be Wrong

**Exact accounting of factual predictions in language models.**

This repository contains code and experiments for studying how a language model produces a factual answer—and how an incorrect answer can arise even when the subject contributes evidence for the correct one.

For a given forward pass, we hold the realized attention patterns, MLP gate values and normalization scales fixed. The resulting affine computation reconstructs the observed activations and logits, allowing us to decompose a target–competitor margin into contributions from the subject, the template, the first token and model biases.

We also compare derivatives through this frozen computation with the model’s live derivatives. This separates the direct effect through the fixed computation (**content**) from the remaining effect through changes in attention, gates and normalization (**selection**).

## Three regimes of factual error

Failures are classified using subject-origin support relative to successful examples of the same relation:

- **Absent:** neither the target nor the competitor meets the calibrated subject-support criteria.
- **Misretrieved:** the competitor meets those criteria and the subject contribution favours it over the target.
- **Overridden:** the target meets those criteria and the subject contribution favours it, but the remaining origins outweigh that contribution.

These are observational categories of the realized computation. They do not establish that a fact is absent from the weights or identify a unique causal stage of failure. Uncalibrated, mixed, undefined-share and continuation failures are retained separately.

## Repository contents

The current experimental code is in [`supplementary/`](supplementary/). Earlier scripts remain at the repository root for the original analyses.

```text
supplementary/
├── notebooks/       # Experiment configuration, launch, monitoring and export
├── suites/          # Readable source and tests for each study
├── results/         # Compact observed results and provenance
├── docs/            # Experiment index, definitions and reproduction details
├── tools/           # Validation and result-export utilities
└── requirements.txt
```

Study versions are kept separate because they include different implementations of modules with the same names. Run each notebook or command from its corresponding suite; do not merge the suites into one Python package.

## Experiments

| Study | Notebook |
|---|---|
| Controlled circuits, source identification and removal baselines | [01 — Controlled and validation](supplementary/notebooks/01_controlled_and_validation.ipynb) |
| Live routing interventions and planted failures | [02 — Causal interventions](supplementary/notebooks/02_causal_interventions.ipynb) |
| Donor-based follow-up experiments | [03 — Donor follow-ups](supplementary/notebooks/03_donor_followups.ipynb) |
| Gate factors and weight edits | [04 — Gate experiments](supplementary/notebooks/04_gate_experiments.ipynb) |
| Pattern edits and gate/edge intervention curves | [05 — Pattern experiments](supplementary/notebooks/05_pattern_experiments.ipynb) |
| Content–selection gradient profiles | [06 — Channels](supplementary/notebooks/06_channels.ipynb) |
| Origin regimes, crossed variance and sink constancy | [07 — Reviewer follow-ups](supplementary/notebooks/07_reviewer_followups.ipynb) |
| Random-initialization/readout nulls, singleton freezes and layer-freeze matrix | [08 — Combined experiment queue](supplementary/notebooks/08_overnight.ipynb) |
| Gradient certificates, calibration sensitivity and anisotropy | [09 — Precision experiments](supplementary/notebooks/09_precision.ipynb) |
| Signed residual-mean projections under singleton freezes | [10 — Signed projection](supplementary/notebooks/10_signed_projection.ipynb) |
| Live/frozen activation adjoints, gate Jacobians, learning steps and curvature | [11 — Adjoints and learning](supplementary/notebooks/11_adjoints.ipynb) |

See the [experiment index](supplementary/docs/EXPERIMENTS.md) for the mapping to manuscript results. Earlier readability-based diagnostics are preserved for historical comparisons and negative results; they are distinct from the current origin-based regimes.

## Running the experiments

### Requirements

Use Python 3.10 or later with a compatible CUDA-enabled PyTorch installation for pretrained-model experiments. The implementation uses **Transformers 4.48.3**; architecture-specific hooks depend on this version. Other dependencies are listed in [`supplementary/requirements.txt`](supplementary/requirements.txt).

Access to the selected Hugging Face models must already be configured. The notebooks contain no authentication credentials or login steps. Experiments use float32; an 8B model requires approximately 32 GB for weights alone, with additional memory needed for computation.

### Workflow

1. Clone the repository and open a notebook from `supplementary/notebooks/` in your GPU environment.
2. Review the model selection, input paths, prompt counts and optional stages in its configuration cells.
3. Choose a new run name, then execute the configuration and launch cells.
4. Use the notebook’s monitor and export cells to inspect progress and save results.

Keep the complete `supplementary/` directory together. The notebooks load the readable source from `suites/` and create versioned working copies.

Later studies include cohort and checkpoint-revision evidence. Some earlier studies require a preceding run’s output or construct a fresh PopQA cohort; their notebooks specify the required inputs. Notebook defaults do not always reproduce the full paper campaign: consult the archived configurations for the reported settings. Optional stages must be enabled before launching.

For detached runs, closing the notebook browser does not stop the worker, but shutting down the GPU instance does. Download results before deleting the instance. Use a new output directory whenever the configuration changes.

## Models and data

The main studies use six factual relations from **PopQA**: author, capital, capital of, composer, director and place of birth. The evaluation cohort contains 599 facts rendered with five templates, giving 2,995 prompts per model.

The model set comprises:

- Llama 3.2 1B and 3B
- Llama 3.1 8B
- Qwen 2.5 3B and 7B
- OLMo-2-1124-7B

Checkpoint revisions and campaign settings are recorded in the included evidence and result metadata. Model weights are not distributed in this repository. Dataset and model use remain subject to their respective licenses and access conditions.

## Results

[`supplementary/results/observed/`](supplementary/results/observed/) contains selected summary tables, figures and run metadata from the channel, reviewer-follow-up, overnight, precision, signed-projection and adjoint studies. These are recorded experimental outputs, not results generated by the implementation tests.

The compact release excludes model weights and large activation/gradient caches. It is sufficient to inspect the included aggregate reports, but is not a complete raw-data archive and cannot resume the original GPU runs or support every per-prompt reanalysis. Earlier studies’ generating code is included even where their results are not part of this subset.

To list the included CSV tables:

```bash
python supplementary/tools/index_results.py
```

To export a compact result archive from a completed run:

```bash
python supplementary/tools/export_results.py /path/to/completed_run \
  --out /path/to/results_summary.zip
```

The export reads the run without modifying it and excludes large caches and per-prompt/per-check report files by default. Use `--include-rows` to include those scalar report tables. The archive records included and skipped files in `EXPORT_MANIFEST.json`.

## Validation

Run the lightweight release checks without downloading model weights:

```bash
python supplementary/tools/check_release.py
python supplementary/tools/test_packaging.py
```

With the scientific dependencies installed, run the implementation tests:

```bash
python supplementary/tools/run_tests.py
```

Or select one suite:

```bash
python supplementary/tools/run_tests.py --suite 11_adjoints
```

The release passed **163 implementation tests across 11 suites**, using small randomly initialized models on CPU. Tests cover reconstruction, gradient calculations, intervention restoration and reporting, among other implementation properties. They do not substitute for the pretrained-model experiments. Details are recorded in the [validation report](supplementary/docs/VALIDATION.md).

## Interpreting the measurements

- **Exact reconstruction is conditional accounting.** The frozen map reconstructs the observed pass; it is not the live model’s response to arbitrary changes.
- **Norm fractions are not additive shares.** Content and selection can oppose one another, and singleton-freeze effects can overlap.
- **Variance is not causal control.** Subject/template variance and crossover measurements describe variation under the experimental design.
- **Numerical certificates have a stated scope.** Per-pass reconstruction checks differ from subsampled finite-difference checks. Unresolved small signals are reported separately.
- **Support is readout-dependent.** The principal decomposition concerns a specified first-token target–competitor margin, rather than an entire generated answer.

See [measurement definitions](supplementary/docs/DEFINITIONS.md) and the [reproduction guide](supplementary/docs/REPRODUCIBILITY.md) for precise conventions, calibration rules and limitations.
