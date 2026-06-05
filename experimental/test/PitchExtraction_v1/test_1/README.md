# PitchExtraction_v1 Test 1

This test runs `PitchExtraction_v1` on a small GTSinger subset:

```text
/mnt/archive/GTSinger/English/EN-Alto-1
```

It does not save `.npy` files. It builds a temporary manifest, runs runtime
feature/label generation, attaches NanoPitch outputs, and writes a JSON summary.

Example:

```bash
/home/DREXEL/pc833/miniconda3/envs/torch_it/bin/python test/PitchExtraction_v1/test_1/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 4 \
  --batch-size 1 \
  --device cuda
```

By default, limited runs use `--selection round_robin`, so `--limit 10` samples
across top-level technique folders instead of taking only the alphabetically
first folder. To reproduce the old behavior, pass `--selection first`.

Outputs are written to:

```text
test/PitchExtraction_v1/test_1/outputs/
  manifest.json
  summary.json
  predictions/*.csv      # if --save-predictions is passed
```
