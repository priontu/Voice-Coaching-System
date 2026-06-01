# PitchExtraction_RMVPE_v1 Test 3

Sweeps four confidence thresholds above the current recommended `0.35`:

- `0.40`
- `0.45`
- `0.50`
- `0.55`

The main comparison is `aggregate_note_median_metrics`, which matches the
recommended coaching score: stable-note median pitch.

Run:

```bash
python test/PitchExtraction_RMVPE_v1/test_3/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 20 \
  --device cuda
```

To save per-note CSVs:

```bash
python test/PitchExtraction_RMVPE_v1/test_3/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 20 \
  --device cuda \
  --save-note-events
```
