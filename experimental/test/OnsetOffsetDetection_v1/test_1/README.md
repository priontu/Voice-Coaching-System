# OnsetOffsetDetection_v1 Test 1

Smoke/evaluation test for the onset/offset Conformer-style model.

Smoke test without a checkpoint:

```bash
python test/OnsetOffsetDetection_v1/test_1/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English \
  --limit 4 \
  --device cuda
```

Evaluate a trained checkpoint:

```bash
python test/OnsetOffsetDetection_v1/test_1/run_test.py \
  --dataset-root /mnt/archive/GTSinger/English \
  --checkpoint OnsetOffsetDetection_v1/runs/english_v1/checkpoints/last.ckpt \
  --limit 100 \
  --device cuda
```

Without `--checkpoint`, metrics are not meaningful; the script only verifies
that data loading, feature generation, model forward pass, and event decoding run.
