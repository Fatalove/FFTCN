# Milestone 7 artifacts

Scope: guided feature-fusion and non-causal TCN practice plus final user validation.

Generated files:

- `models/merge/sleep_edf_fusion_tcn.py`: formal user exercise skeleton.
- `tests/test_sleep_edf_fusion_tcn.py`: behavior-focused padding, non-causality, residual, dilation, fusion, and gradient tests.
- `learning_guides/milestone_07/README.md`: ACM-style Chinese learning guide.
- `learning_guides/milestone_07/reference_solution.py`: independent annotated reference answer.
- `reproduction_artifacts/milestone_07/validation_summary.json`: setup validation evidence.

Final status:

- The user completed the formal implementation. All 6 milestone-focused tests and all 26 project regression tests pass.
- Tests verify behavior and experiment semantics rather than arbitrary internal attribute names.
- The reference solution reuses the accepted milestone 5/6 feature branches and is not imported by formal code.
- Both the reference and user implementations match the repository full-model parameter count (`1,189,371`) and TCN parameter count (`587,904`).
- User validation confirms same-padding length preservation, non-causal future visibility, residual projection, dilation order `[1,1,2,2,4,4,8,8]`, `[B,T,472] -> [B,T,128] -> [B,T,5]`, and gradients through both feature branches and the TCN.
- The teaching interface preserves `[B,T,5]`; the repository flattens the same positions to `[B*T,5]`.
- Oversampling is intentionally outside this forward-model milestone. Repository `offset_resample` is reserved for raw/wave training-set pretraining with `seq_len=1`; it is not applied to validation, test, or `seq_len=50` merge fine-tuning. This boundary is covered in milestone 8.
