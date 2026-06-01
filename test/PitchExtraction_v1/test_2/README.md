# PitchExtraction_v1 Test 2

This test evaluates `PitchExtraction_v1` on **Control_Group only** recordings
from:

```text
/mnt/archive/GTSinger/English/EN-Alto-1
```

It scans all technique/song folders, but only keeps recordings whose immediate
parent folder is:

```text
Control_Group
```

Run:

```bash
/home/DREXEL/pc833/miniconda3/envs/torch_it/bin/python test/PitchExtraction_v1/test_2/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 20 \
  --batch-size 1 \
  --device cuda \
  --save-predictions
```

Outputs:

```text
test/PitchExtraction_v1/test_2/outputs/
  manifest.json
  summary.json
  predictions/*.csv
```

Limited runs default to `--selection round_robin`, so samples are spread across
top-level technique folders while still using only `Control_Group` recordings.

