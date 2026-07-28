# Milestone 5 artifacts

Scope: guided 1D-CNN temporal-branch practice setup.

Generated files:

- `models/raw/sleep_edf_1d_cnn.py`: formal user exercise skeleton.
- `tests/test_sleep_edf_1d_cnn.py`: focused shape, mode-switch, and gradient tests.
- `learning_guides/milestone_05/README.md`: ACM-style Chinese learning guide.
- `learning_guides/milestone_05/reference_solution.py`: independent annotated reference answer.
- `reproduction_artifacts/milestone_05/validation_summary.json`: setup validation evidence.

Validation status:

- Completed on 2026-07-13 after the user's implementation passed behavior-focused validation.
- The independent reference solution is not imported by formal code.
- Reference validation covers `[B,1,3000] -> [B,256]`, `[B,1,3000] -> [B,5]`, output-mode independence from `train()/eval()`, and finite gradients in the first convolution.
- The reference and repository `RawFeatureNet` each contain `402,447` parameters and produce matching output shapes in both modes.
- The focused suite passed 5/5 tests and the milestone 3/4 regression suite passed 8/8 tests.
- The test was adjusted to inspect module behavior and ordering rather than require particular internal attribute names; the user's core implementation was not changed.
