# Note Onset / Offset Detector

CNN + BiLSTM model for detecting note boundary timing from singing audio.  
Part of the VocalCoach interpretable singing evaluation system.

---

## What this does

```
WAV audio
    → log-mel spectrogram
    → CNN + BiLSTM
    → onset probability curve  [T]
    → offset probability curve [T]
    → peak picking
    → note boundaries: [{onset_time, offset_time, duration}, ...]
    → coaching metrics: OnsetError, OffsetError, DurationError
```

The model does **not** transcribe pitch. It detects only **timing boundaries**.

---

## Project Structure

```
Note Model/
├── configs/
│   └── default.yaml          ← all hyperparameters live here
│
├── model.py                  ← OnsetOffsetModel (CNN → BiLSTM → 2 MLP heads)
├── utils.py                  ← audio I/O, spectrogram, Gaussian labels, peak picking
├── dataset.py                ← NoteDetectionDataset + DataLoader factory
├── prepare_gtsinger.py       ← GTSinger → train/val/test manifests
├── train.py                  ← training loop, early stopping, checkpoints
├── inference.py              ← NoteDetector class + CLI
├── metrics.py                ← P/R/F1, onset/offset/duration MAE
├── visualization.py          ← waveform, spectrogram, probability plots
└── README.md
```

---

## Dependencies

### Python version

Python 3.10 or later is required.

### Install

```bash
pip install torch torchaudio librosa numpy scipy matplotlib pyyaml tqdm music21
```

Full pinned list:

| Package    | Purpose                              |
|------------|--------------------------------------|
| torch      | model, training, GPU                 |
| torchaudio | audio I/O, mel spectrogram, SpecAugment |
| librosa    | audio utilities (optional fallback)  |
| numpy      | array ops                            |
| scipy      | peak picking (`find_peaks`)          |
| matplotlib | visualization                        |
| pyyaml     | loading `configs/default.yaml`       |
| tqdm       | training progress bars               |
| music21    | MusicXML parsing (optional)          |

> **GPU** — install the CUDA build of PyTorch from https://pytorch.org/get-started.  
> The code falls back to CPU automatically if no GPU is available.

---

## GTSinger Dataset

### Where it is on your machine

```
C:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Phoneme Model\gtsinger\
```

### Dataset structure

```
gtsinger/
└── English/
    └── EN-Alto-1/
        ├── Breathy/
        │   ├── all is found/
        │   │   ├── Breathy_Group/
        │   │   │   ├── 0000.wav          ← singing audio
        │   │   │   ├── 0000.json         ← note timings (used by this model)
        │   │   │   ├── 0000.musicxml     ← MusicXML (also supported)
        │   │   │   └── 0000.TextGrid     ← Praat phoneme labels (not used here)
        │   │   ├── Control_Group/
        │   │   └── Paired_Speech_Group/
        │   ├── god is a girl/
        │   ├── innocence/
        │   ├── let it go/
        │   ├── once up a dream/
        │   ├── safe and sound/
        │   ├── sleepyhead/
        │   ├── someday or oneday/
        │   ├── treasure/
        │   └── who says/
        ├── Glissando/
        └── Mixed_Voice_and_Falsetto/
```

### Singing techniques

| Folder name              | Description                    |
|--------------------------|--------------------------------|
| `Breathy`                | Breathy vocal quality          |
| `Glissando`              | Pitch glide between notes      |
| `Mixed_Voice_and_Falsetto` | Mixed head/chest voice       |

### Recording groups

| Group name           | Description                                      |
|----------------------|--------------------------------------------------|
| `Breathy_Group`      | Target technique applied                         |
| `Control_Group`      | Same song without the technique (normal singing) |
| `Paired_Speech_Group`| Same words spoken (not sung)                     |

### GTSinger JSON format

Each `.json` file is a list of word-level entries. Note timings are stored as lists because one word can span multiple notes (melisma):

```json
[
  {
    "word": "let",
    "start_time": 1.59,
    "end_time": 1.71,
    "note": [52],
    "note_start": [1.59],
    "note_end":   [1.71],
    "ph": ["L", "EH1", "T"],
    "breathy": ["1"],
    "vibrato": ["0"],
    ...
  },
  {
    "word": "it",
    "note": [54, 56],
    "note_start": [1.71, 2.075],
    "note_end":   [2.075, 2.31],
    ...
  }
]
```

The model reads `note_start` and `note_end` directly. Silence entries  
(`word == "<AP>"` or `note == 0`) are automatically skipped.

---

## Quick Start

### Step 1 — Prepare manifests

Run once to build train/val/test split files from your GTSinger dataset:

```bash
cd "c:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Note Model"

python prepare_gtsinger.py \
    --gtsinger_root "C:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Phoneme Model\gtsinger" \
    --output_dir    data/manifests \
    --language      English \
    --technique     ALL \
    --group         Breathy_Group \
    --train_ratio   0.8 \
    --val_ratio     0.1 \
    --seed          42
```

This writes:

```
data/manifests/
    train.json          ← 80 % of songs
    val.json            ← 10 % of songs
    test.json           ← 10 % of songs
    dataset_stats.txt   ← breakdown by technique / group / song
```

**Technique options:** `Breathy` | `Glissando` | `Mixed_Voice_and_Falsetto` | `ALL`  
**Group options:** `Breathy_Group` | `Control_Group` | `Paired_Speech_Group` | `ALL`

---

### Step 2 — Train

```bash
python train.py \
    --config         configs/default.yaml \
    --train_manifest data/manifests/train.json \
    --val_manifest   data/manifests/val.json
```

Checkpoints are saved to `checkpoints/` by default.

**Optional flags:**

```bash
# Resume from a checkpoint
python train.py ... --resume checkpoints/epoch_050.pt

# Save checkpoints elsewhere
python train.py ... --checkpoint_dir runs/experiment_01

# Specify number of epochs
python train.py ... --epochs 20
```

Console output per epoch:
```
09:12:34  INFO     Epoch 001 | tr_loss=0.1823 | va_loss=0.1654 | on_f1=0.412 | off_f1=0.389 | lr=1.00e-03
09:12:51  INFO     Epoch 002 | tr_loss=0.1541 | va_loss=0.1432 | on_f1=0.511 | off_f1=0.478 | lr=1.00e-03
...
09:14:22  INFO       checkpoint → checkpoints/best.pt
```

---

### Step 3 — Run inference

```bash
python inference.py \
    --checkpoint checkpoints/best.pt \
    --config     configs/default.yaml \
    --audio      path/to/singing.wav \
    --output     results.json
```

**Output** (`results.json`):

```json
[
  {"onset_time": 1.59,  "offset_time": 1.71,  "duration": 0.12},
  {"onset_time": 1.71,  "offset_time": 2.31,  "duration": 0.60},
  {"onset_time": 2.45,  "offset_time": 3.10,  "duration": 0.65}
]
```

Without `--output` the JSON is printed to stdout.

---

### Step 4 — Evaluate against ground truth

```powershell
py .\evaluate.py `
    --checkpoint checkpoints/best.pt `
    --config     configs/default.yaml `
    --audio      "C:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Phoneme Model\gtsinger\English\EN-Alto-1\Breathy\let it go\Breathy_Group\0000.wav" `
    --label      "C:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Phoneme Model\gtsinger\English\EN-Alto-1\Breathy\let it go\Breathy_Group\0000.json"
```

**Whole test split:**

```powershell
py .\evaluate.py `
    --checkpoint checkpoints/best.pt `
    --config     configs/default.yaml `
    --manifest   data/manifests/test.json `
    --output     eval_results.json
```

**Optional flags:**

```powershell
# Tighter timing tolerance (default is 50 ms)
py .\evaluate.py ... --tolerance_ms 30
```

**Console output:**

```
────────────────────────────────────────────
  Average over 14 files
────────────────────────────────────────────
  Metric                       Value
  ──────                       ─────
  onset_precision              0.8700
  onset_recall                 0.8300
  onset_f1                     0.8500
  onset_mae_ms                 18.4000
  offset_precision             0.8400
  offset_recall                0.8000
  offset_f1                    0.8200
  offset_mae_ms                22.1000
  duration_mae_ms              31.5000
  relative_duration_mae        0.0900
  n_matched_notes              42.0000
────────────────────────────────────────────
```

---

### Step 5 — Visualize

All plots are saved as PNG files to the `plots/` folder (or wherever you point `--output_dir`).

**Four-panel overview** (waveform + spectrogram + onset curve + offset curve):

```powershell
py .\visualize.py `
    --checkpoint checkpoints/best.pt `
    --config     configs/default.yaml `
    --audio      "C:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Phoneme Model\gtsinger\English\EN-Alto-1\Breathy\let it go\Breathy_Group\0000.wav" `
    --plot       overview
```

**Overlay ground truth boundaries** (adds coloured dashed lines for reference):

```powershell
py .\visualize.py `
    --checkpoint checkpoints/best.pt `
    --config     configs/default.yaml `
    --audio      "...\Breathy_Group\0000.wav" `
    --label      "...\Breathy_Group\0000.json" `
    --plot       all `
    --output_dir plots/0000
```

**`--plot` options:**

| Value         | Output file            | What it shows                                      |
|---------------|------------------------|----------------------------------------------------|
| `overview`    | `*_overview.png`       | All four panels stacked — best starting point      |
| `waveform`    | `*_waveform.png`       | Audio waveform with onset / offset markers         |
| `spectrogram` | `*_spectrogram.png`    | Log-mel spectrogram with boundary overlays         |
| `probs`       | `*_probs.png`          | Onset and offset probability curves with threshold |
| `all`         | All four files above   | Everything at once                                 |

---

## Configuration Reference

All settings are in `configs/default.yaml`. Change values there — no code edits needed.

### Audio

| Key           | Default | Description                          |
|---------------|---------|--------------------------------------|
| `sample_rate` | 16000   | Resampling target (Hz)               |
| `n_fft`       | 1024    | STFT window size                     |
| `hop_length`  | 256     | Hop size → 16ms per frame at 16kHz   |
| `n_mels`      | 80      | Mel filterbank bins                  |
| `fmin`        | 0.0     | Lowest mel frequency                 |
| `fmax`        | 8000.0  | Highest mel frequency                |

### Model

| Key                | Default        | Description                       |
|--------------------|----------------|-----------------------------------|
| `cnn_channels`     | [32,64,128,128]| Output channels per ConvBlock     |
| `lstm_hidden_size` | 256            | BiLSTM hidden units per direction |
| `lstm_num_layers`  | 2              | Stacked BiLSTM layers             |
| `lstm_dropout`     | 0.3            | Dropout between LSTM layers       |
| `head_hidden_size` | 128            | MLP head intermediate size        |
| `dropout`          | 0.3            | General dropout rate              |

### Training

| Key             | Default | Description                             |
|-----------------|---------|-----------------------------------------|
| `batch_size`    | 16      | Samples per batch                       |
| `num_epochs`    | 100     | Maximum training epochs                 |
| `learning_rate` | 0.001   | Adam initial LR                         |
| `weight_decay`  | 0.0001  | L2 regularization                       |
| `patience`      | 15      | Early stopping patience (epochs)        |
| `grad_clip`     | 1.0     | Gradient norm clipping                  |
| `label_sigma`   | 0.02    | Gaussian label sigma in seconds (~1 frame) |

### Peak Picking

| Key                   | Default | Description                           |
|-----------------------|---------|---------------------------------------|
| `onset_threshold`     | 0.3     | Min probability to be a peak          |
| `offset_threshold`    | 0.3     | Min probability to be a peak          |
| `min_distance_frames` | 3       | Minimum frames between consecutive peaks |

Tune these after training using your validation set. Lower threshold = more detections (higher recall, lower precision).

---

## Manifest Format

Both `prepare_gtsinger.py` and the dataset expect this simple format:

```json
[
  {
    "audio": "C:/path/to/0000.wav",
    "label": "C:/path/to/0000.json"
  },
  ...
]
```

The label file can be:
- **GTSinger JSON** (has `note_start` key) — detected automatically
- **Simple JSON** (`[{"onset": 1.2, "offset": 1.8}, ...]`)
- **MusicXML** (`.xml` or `.musicxml`)

---

## Programmatic API Reference

### Load audio

```python
from utils import load_audio, compute_log_mel_spectrogram, normalize_spectrogram

waveform, sr = load_audio("singing.wav", target_sr=16000)
log_mel = compute_log_mel_spectrogram(waveform, sample_rate=16000)
log_mel = normalize_spectrogram(log_mel)  # [1, 80, T]
```

### Parse GTSinger annotations

```python
from utils import parse_gtsinger_json

notes = parse_gtsinger_json("0000.json")
for n in notes:
    print(f"onset={n.onset:.3f}s  offset={n.offset:.3f}s  dur={n.duration:.3f}s")
```

### Build Gaussian labels

```python
from utils import build_onset_labels, build_offset_labels, frames_to_time
import numpy as np

n_frames = log_mel.shape[-1]
frame_times = frames_to_time(n_frames, hop_length=256, sample_rate=16000)

onset_labels  = build_onset_labels(notes, frame_times, sigma=0.02)
offset_labels = build_offset_labels(notes, frame_times, sigma=0.02)
```

### Run model forward pass

```python
import torch
from model import OnsetOffsetModel

model = OnsetOffsetModel()
x = log_mel.unsqueeze(0)           # [1, 1, 80, T]
onset_logits, offset_logits = model(x)
onset_probs, offset_probs   = model.predict(x)   # sigmoid, no_grad
```

### Dataset + DataLoader

```python
from dataset import NoteDetectionDataset, create_dataloaders

# From a manifest file
ds = NoteDetectionDataset(
    manifest_path="data/manifests/train.json",
    segment_length=10.0,   # seconds
    augment=True,
)

# From a directory (auto-discovers audio/ and labels/ subdirs)
ds = NoteDetectionDataset(data_dir="my_data/")

# Ready-made train + val loaders
train_loader, val_loader = create_dataloaders(
    train_manifest="data/manifests/train.json",
    val_manifest="data/manifests/val.json",
    batch_size=16,
)
```

### NoteDetector inference

```python
from inference import NoteDetector

detector = NoteDetector.from_checkpoint(
    checkpoint_path="checkpoints/best.pt",
    config_path="configs/default.yaml",
)

notes = detector.detect("singing.wav")
# [{"onset_time": 1.23, "offset_time": 1.57, "duration": 0.34}, ...]

on_probs, off_probs, frame_times = detector.predict_probs("singing.wav")
```

---

## Model Architecture

```
Input: Log-Mel Spectrogram  [B, 1, 80, T]
           │
   ┌───────▼────────┐
   │  ConvBlock ×4  │  Conv2d → BatchNorm → ReLU → MaxPool2d(2,1)
   │                │  Frequency: 80 → 40 → 20 → 10 → 5
   │                │  Time:      T  unchanged
   └───────┬────────┘
           │  [B, 128×5=640, T]
           │  permute → [B, T, 640]
   ┌───────▼────────┐
   │  BiLSTM ×2     │  hidden=256, bidirectional → output=512
   └───────┬────────┘
     ┌─────┴──────┐
     ▼            ▼
  onset head   offset head
  Linear(512→128)→ReLU→Linear(128→1)
     │            │
  [B, T]       [B, T]   (raw logits, apply sigmoid for probabilities)
```

Frame rate: `hop_length / sample_rate = 256 / 16000 = 16 ms per frame`

---

## Tips & Tuning

**If onset F1 is low:**
- Lower `onset_threshold` in `default.yaml` (try 0.2)
- Increase `label_sigma` to widen the Gaussian target (try 0.03)
- Add more training data from `Control_Group` using `--group ALL`

**If too many false detections:**
- Raise `onset_threshold` (try 0.4–0.5)
- Increase `min_distance_frames` (try 5)

**If training is slow:**
- Reduce `segment_length` from 10.0 to 6.0 seconds
- Reduce `batch_size` if OOM on GPU

**For a larger model:**
- Increase `lstm_hidden_size` to 512
- Add another entry to `cnn_channels`: `[32, 64, 128, 128, 256]`

---

## File Responsibilities

| File                  | What to change it for                                    |
|-----------------------|----------------------------------------------------------|
| `configs/default.yaml`| Hyperparameters — always start here                      |
| `model.py`            | Architecture changes                                     |
| `utils.py`            | Audio processing or label generation changes             |
| `dataset.py`          | Data loading, augmentation, new annotation formats       |
| `prepare_gtsinger.py` | Adding new languages, techniques, or split strategies    |
| `train.py`            | Optimizer, scheduler, loss weighting                     |
| `inference.py`        | Post-processing, batch inference over multiple files     |
| `metrics.py`          | New evaluation metrics or tolerance windows              |
| `visualization.py`    | New plot types                                           |
