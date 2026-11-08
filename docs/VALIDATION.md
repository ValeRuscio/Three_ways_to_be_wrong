# Release validation

Validated 8 November 2026. **163 implementation tests passed across 11 isolated suites.**

| Suite | Tests passed |
|---|---:|
| 01_controlled_and_validation | 24 |
| 02_causal_interventions | 22 |
| 03_donor_followups | 13 |
| 04_gate_experiments | 15 |
| 05_pattern_experiments | 34 |
| 06_channels | 7 |
| 07_reviewer_followups | 8 |
| 08_overnight | 12 |
| 09_precision | 17 |
| 10_signed_projection | 5 |
| 11_adjoints | 6 |

Additional checks passed: Python/notebook syntax; all 11 notebook repository-source and checksum cells; deterministic source packaging; compact-export file selection, metadata redaction and overwrite protection. Run the packaging tests with `python supplementary/tools/test_packaging.py` from the repository root.

Environment versions are recorded in `validation_environment.json`; per-suite exit status and elapsed time are in `validation_tests.json`. Tests used small random models on CPU, with Hugging Face offline mode. They do not rerun pretrained GPU experiments, validate every manuscript number or establish causal interpretations. The legacy original suite is included for provenance but was not separately rerun; the 11 subsequent suites were tested.

Notebook configuration, detached CUDA launch and full pretrained execution were not executed during packaging. Scientific source was extracted from the previously supplied notebook bundles; packaging changes remove embedded payloads and installation cells, expose source, and add documentation/export tools.
