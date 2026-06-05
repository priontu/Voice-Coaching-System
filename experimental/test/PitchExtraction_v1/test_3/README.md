# PitchExtraction_v1 Test 3

Diagnostic plot test for NanoPitch on GTSinger `Control_Group` recordings.

This test is meant to answer:

```text
Is NanoPitch failing, or is the MusicXML-vs-audio timing alignment bad?
```

It runs NanoPitch, builds a MusicXML-derived reference F0 contour, saves both
contours to CSV, estimates the best global time lag, and writes PNG plots.

Run:

```bash
/home/DREXEL/pc833/miniconda3/envs/torch_it/bin/python test/PitchExtraction_v1/test_3/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English/EN-Alto-1 \
  --limit 3 \
  --device cuda
```

Outputs:

```text
test/PitchExtraction_v1/test_3/outputs/
  manifest.json
  summary.json
  contours/*.csv
  plots/*.png
```

The plot shows:

```text
NanoPitch F0
MusicXML reference F0
MusicXML reference shifted by estimated best lag
```

