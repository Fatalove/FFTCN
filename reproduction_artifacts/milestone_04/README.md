# Milestone 4 artifacts

Scope: Morlet CWT practice setup.

Generated files:

- `data/sleep_edf_cwt.py`: user exercise skeleton for `morlet_cwt_epoch`.
- `tests/test_sleep_edf_cwt.py`: focused tests for shape, dtype, finite values, determinism, and alignment with repository Torch CWT.
- `learning_guides/milestone_04/README.md`: ACM-style Chinese guide.
- `learning_guides/milestone_04/reference_solution.py`: complete annotated reference answer, not imported by formal code.

Validation status:

- Completed on 2026-07-13 after review of the user's formal implementation.
- The focused CWT suite passed 4/4 tests; the preprocessing plus CWT regression suite passed 8/8 tests.
- Fixed random input numerically aligns with `data.wavelet_torch.cwt`: maximum absolute error `8.642673492431641e-07`, within `rtol=atol=1e-5`.
- The formal implementation intentionally differs from the original Torch implementation for flat all-zero/constant inputs: it returns a finite zero image instead of NaN because the milestone contract requires no NaN/Inf.
- Codex did not modify the user's core implementation during acceptance.
