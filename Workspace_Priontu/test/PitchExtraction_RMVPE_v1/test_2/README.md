# PitchExtraction_RMVPE_v1 Test 2

Compares RMVPE scoring/post-processing strategies without retraining:

- confidence threshold sweeps
- median smoothing
- short-gap interpolation
- large-jump deglitching
- stable note interiors
- note-level median pitch
- global/local lag diagnostics

Run:

```bash
python test/PitchExtraction_RMVPE_v1/test_2/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 20 \
  --device cuda
```

The output summary contains `aggregate_frame_metrics` and
`aggregate_note_metrics` so you can compare which strategy improves the score.
