# Milestone 8 artifacts

Scope: two-stage FFTCN training strategy, record-local offset oversampling, differential learning rates, and portable checkpoints.

Generated files:

- `training/sleep_edf_two_stage.py`: formal user exercise skeleton.
- `tests/test_sleep_edf_two_stage.py`: behavior-focused tests for data isolation, oversampling, training/validation behavior, feature transfer, optimizer groups, and checkpoint round trips.
- `learning_guides/milestone_08/README.md`: staged Chinese tutorial with a manual example and fully annotated core implementation.
- `learning_guides/milestone_08/reference_solution.py`: independent reference implementation.
- `reproduction_artifacts/milestone_08/validation_summary.json`: setup validation evidence.

Current status:

- The user completed every formal exercise interface; no `NotImplementedError` remains.
- The formal implementation passes all 6 milestone-focused behavior tests and all 32 milestone 2-8 regression tests.
- A real-data smoke check used two epochs from `SC4001.npz`, generated CWT inputs, updated both pretrained branches once, transferred their weights, and updated the fusion model once; all three losses were finite.
- The reference solution passes all 6 milestone-focused behavior tests.
- The previous milestone suite passes all 26 regression tests.
- Oversampling is enabled only for raw/wave `seq_len=1` training-set pretraining; validation, test, and `seq_len=50` fusion fine-tuning remain unbalanced.
- Repository-style offset oversampling retains originals and adds `n_max` shifted windows per existing class, so it reduces imbalance without strictly equalizing counts.
- Fine-tuning uses the base learning rate for the new TCN/classifier and `base_lr * scale` for both pretrained feature branches; the branches remain trainable.
- Checkpoints use `state_dict` plus configuration and label metadata instead of serializing the complete Python model object.
