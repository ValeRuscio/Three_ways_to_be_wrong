# Definitions that must remain consistent with the paper

- The frozen program fixes realized attention probabilities, SiLU outputs and normalization scales. Its affine replay reconstructs the baseline pass; it is not the original transformer's live input-output function.
- Origin contributions are centred separately across vocabulary. A token's subject share is **signed** centred subject support divided by the sum of absolute centred origin contributions. The denominator may be zero.
- Source presence requires centred support and share to exceed their success-calibrated thresholds **and zero**. Calibration is subject-disjoint; cross-fit calibration is a separately labelled sensitivity analysis.
- Uncalibrated, undefined, mixed and continuation cases remain explicit. Absent origin support does not mean no stored knowledge or literally zero target support.
- Full gradient split: content = all-frozen derivative; selection = live minus all-frozen. The reported fraction `||selection||/(||selection||+||content||)` is a norm ratio, not an additive share of live sensitivity.
- Singleton split for freeze M: `H_M = G_freeze_M`, `D_M = G_live-H_M`. Its fraction is `||D_M||/(||D_M||+||H_M||)` and its cosine is `cos(H_M,D_M)`. It is not a fraction of the full selection norm removed. Singleton effects overlap; their nonadditivity is not proof of three independent compensation mechanisms.
- Attention ANOVA compares semantic role masses, not raw token rows of unequal lengths. Variance dominance and a first crossing are descriptive, not a validated causal hand-off.
- The activation-adjoint final cut is before the final normalization; live and frozen adjoints need not agree there.
- Finite-change linearity applies to one independently varied, untied matrix with the original program frozen. Simultaneous updates can have cross-matrix curvature.
- A finite-difference test can fail at a finite epsilon without invalidating the derivative identity. Unresolved small signals are not successful independent checks.
- Bootstrap intervals are pointwise and subject-clustered where specified. They are not blanket Holm-adjusted or uncertainty over model training seeds.
