# OnsetOffsetDetection_v1

Conformer-style onset/offset detector for GTSinger English.

The model uses:

- PyTorch modules for a compact Conformer-style encoder
- PyTorch Lightning for training
- torchaudio GPU-side log-mel extraction
- GTSinger `.json` note starts/ends as aligned labels

Outputs:

- `onset_logits`
- `offset_logits`
- `active_logits`

Train on the English subset:

```bash
python OnsetOffsetDetection_v1/train.py \
  --dataset-root /mnt/archive/GTSinger/English \
  --output-dir OnsetOffsetDetection_v1/runs/english_v1 \
  --batch-size 4 \
  --num-workers 4 \
  --max-epochs 20 \
  --accelerator gpu \
  --devices 1
```

Quick smoke train:

```bash
python OnsetOffsetDetection_v1/train.py \
  --dataset-root /mnt/archive/GTSinger/English \
  --output-dir OnsetOffsetDetection_v1/runs/smoke \
  --limit 12 \
  --batch-size 2 \
  --num-workers 0 \
  --max-epochs 1 \
  --accelerator cpu \
  --precision 32
```

Evaluate:

```bash
python OnsetOffsetDetection_v1/evaluate.py \
  --dataset-root /mnt/archive/GTSinger/English \
  --checkpoint OnsetOffsetDetection_v1/runs/english_v1/checkpoints/last.ckpt \
  --output-dir OnsetOffsetDetection_v1/eval_outputs/english_v1 \
  --device cuda
```

The event metrics use peak picking with a default `50 ms` matching tolerance.
