# Phoneme Boundary Detector

Wav2Vec2 + CTC model for detecting phoneme boundary timing from singing audio.  
Part of the VocalCoach interpretable singing evaluation system.

---

## What this does

```
WAV audio
    → load & resample to 16 kHz mono
    → Wav2Vec2 feature encoder + Transformer
    → frame-level logits  [T, vocab_size]
    → CTC greedy argmax
    → collapse repeats → remove blanks
    → frame index × frame_duration → seconds
    → blank-region scan (recover consonants suppressed by blank dominance)
    → phoneme segments: [{phoneme, start_time, end_time, confidence}, ...]
    → coaching metrics: BoundaryPrecision, BoundaryRecall, BoundaryF1, MAE
```

The model does **not** transcribe text. It detects only **phoneme boundary timing**.

---

## Project Structure

```
Phoneme Model/
├── phoneme_model.py      ← PhonemeBoundaryConfig, full pipeline, CTC alignment, CLI
├── evaluate.py           ← batch evaluation against GTSinger TextGrid files
├── examples.py           ← usage examples
├── test.py               ← unit tests (pytest)
├── requirements.txt      ← pinned dependencies
├── QUICKSTART.md         ← 5-minute quickstart
└── README.md
```

---

## Dependencies

### Python version

Python 3.10 or later is required.

### Install

```bash
pip install torch torchaudio transformers soundfile numpy matplotlib pandas textgrid
```

For GPU (CUDA 11.8):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers soundfile numpy matplotlib pandas textgrid
```

Full pinned list:

| Package        | Purpose                                         |
|----------------|-------------------------------------------------|
| torch          | model, inference, GPU                           |
| torchaudio     | audio I/O, resampling                           |
| transformers   | Wav2Vec2 model + `Wav2Vec2PhonemeCTCTokenizer`  |
| soundfile      | audio fallback when FFmpeg is missing (Windows) |
| numpy          | array ops                                       |
| matplotlib     | visualization                                   |
| pandas         | aggregate results CSV (required for `--analyze`)|
| textgrid       | parsing Praat `.TextGrid` reference files       |

> **GPU** — install the CUDA build of PyTorch from https://pytorch.org/get-started.  
> The code falls back to CPU automatically if no GPU is available.

### eSpeak NG (Windows only)

`Wav2Vec2PhonemeCTCTokenizer` requires eSpeak NG on Windows. The module sets these paths automatically:

```python
os.environ.setdefault("PHONEMIZER_ESPEAK_PATH",    r"C:\Program Files\eSpeak NG\espeak-ng.exe")
os.environ.setdefault("ESPEAKNG_PATH",              r"C:\Program Files\eSpeak NG")
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY",  r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
```

Download: https://github.com/espeak-ng/espeak-ng/releases

### Pre-download model (~360 MB)

Run once to cache the model weights locally:

```python
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2PhonemeCTCTokenizer, AutoModelForCTC

Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
```

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
        │   │   │   ├── 0000.TextGrid     ← Praat phoneme labels (used by this model)
        │   │   │   ├── 0000.json         ← note timings (used by Note Model)
        │   │   │   └── 0000.musicxml     ← MusicXML (not used here)
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

| Folder name                | Description                    |
|----------------------------|--------------------------------|
| `Breathy`                  | Breathy vocal quality          |
| `Glissando`                | Pitch glide between notes      |
| `Mixed_Voice_and_Falsetto` | Mixed head/chest voice         |

### Recording groups

| Group name            | Description                                      |
|-----------------------|--------------------------------------------------|
| `Breathy_Group`       | Target technique applied                         |
| `Control_Group`       | Same song without the technique (normal singing) |
| `Paired_Speech_Group` | Same words spoken (not sung)                     |

### TextGrid format

Each `.TextGrid` file is a Praat annotation with a `phone` tier. Each interval has a phoneme label and time boundaries:

```
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 7.32
tiers? <exists>
size = 1
item [1]:
    class = "IntervalTier"
    name = "phone"
    intervals [1]:
        xmin = 0.0
        xmax = 0.08
        text = "sil"
    intervals [2]:
        xmin = 0.08
        xmax = 0.16
        text = "l"
    intervals [3]:
        xmin = 0.16
        xmax = 0.26
        text = "ɛ"
    ...
```

Silence entries (`text == "sil"`, `"pau"`, `""`, or `"<SP>"`) are automatically skipped during evaluation.

---

## Quick Start

### Step 1 — Run inference on a single file

```powershell
cd "c:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Phoneme Model"

py .\phoneme_model.py `
    "C:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Phoneme Model\gtsinger\English\EN-Alto-1\Breathy\let it go\Breathy_Group\0000.wav" `
    --output results.json
```

**Console output:**

```
======================================================================
RESULTS
======================================================================

Phoneme sequence (first 20): l ɛ t ɪ t ɡ oʊ ...

First 5 segments:
  l         0.080s – 0.160s  conf=0.942
  ɛ         0.160s – 0.260s  conf=0.887
  t         0.260s – 0.340s  conf=0.923
  ɪ         0.340s – 0.420s  conf=0.891
  t         0.420s – 0.480s  conf=0.905
```

**Output** (`results.json`):

```json
{
    "phonemes": ["l", "ɛ", "t", "ɪ", "t", "ɡ", "oʊ"],
    "boundaries": [[0.08, 0.16], [0.16, 0.26]],
    "segments": [
        {"phoneme": "l", "start_time": 0.08, "end_time": 0.16, "confidence": 0.9423, "frame_start": 4, "frame_end": 8}
    ],
    "metadata": {
        "audio_file": "0000.wav",
        "duration_s": 7.32,
        "num_phonemes": 17,
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "device": "cuda"
    }
}
```

Without `--output` the JSON is printed to stdout.

**Optional flags:**

```powershell
# Save JSON + visualize (saves <stem>_plot.png)
py .\phoneme_model.py audio.wav --output results.json --plot

# Word-level grouping
py .\phoneme_model.py audio.wav --words

# Force CPU
py .\phoneme_model.py audio.wav --device cpu

# Evaluate against a reference JSON (must have a "segments" key)
py .\phoneme_model.py audio.wav --reference ground_truth.json
```

The `--reference` flag expects a JSON file in the same format as `--output` (with a `"segments"` key). To evaluate against GTSinger `.TextGrid` files, use `evaluate.py` (Step 2).

---

### Step 2 — Batch evaluate over GTSinger

```powershell
py .\evaluate.py `
    --gtsinger-dir "C:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Phoneme Model\gtsinger\English" `
    --device cuda `
    --output eval_results.json
```

**Optional flags:**

```powershell
# Limit files for quick sanity check
py .\evaluate.py --gtsinger-dir .\gtsinger\English --max-files 10

# Tighter timing tolerance (default is 50 ms)
py .\evaluate.py --gtsinger-dir .\gtsinger\English --tolerance-ms 30

# Print per-file metrics as they run
py .\evaluate.py --gtsinger-dir .\gtsinger\English --verbose

# Run analysis after evaluation (saves CSV + 4-panel PNG)
py .\evaluate.py --gtsinger-dir .\gtsinger\English --analyze
```

**Console output:**

```
======================================================================
GTSinger BATCH EVALUATION
======================================================================
Total files:  10
Device:       cuda
Tolerance:    50.0ms
Output:       eval_results.json
======================================================================

[  1/10] 0000.wav                  [+  ] F1=0.6827
[  2/10] 0001.wav                  [~  ] F1=0.5903
[  3/10] 0002.wav                  [+  ] F1=0.6364
...

======================================================================
EVALUATION SUMMARY
======================================================================
Successful:   10/10
Failed:       0/10

Aggregate Metrics:
  Avg Precision: 0.7075
  Avg Recall:    0.6000
  Avg F1-Score:  0.6466
  Median F1:     0.6182
  Std F1:        0.0648
  Min F1:        0.5806
  Max F1:        0.7568
  Avg MAE:       26.73ms
======================================================================

[OK] Results saved to eval_results.json
```

**F1 status indicators:**

| Symbol  | F1 range        |
|---------|-----------------|
| `[+++]` | > 0.85          |
| `[++ ]` | 0.75 – 0.85     |
| `[+  ]` | 0.65 – 0.75     |
| `[~  ]` | 0.50 – 0.65     |
| `[---]` | ≤ 0.50          |

---

### Step 3 — Visualize

```powershell
py .\phoneme_model.py `
    "C:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Phoneme Model\gtsinger\English\EN-Alto-1\Breathy\let it go\Breathy_Group\0000.wav" `
    --output results.json `
    --plot
```

Saves `results_plot.png` (or `phoneme_plot.png` when `--output` is omitted) with two panels:
- **Top panel**: waveform amplitude over time
- **Bottom panel**: colour-coded phoneme segments with IPA labels at midpoints

---

## Configuration Reference

`PhonemeBoundaryConfig` is a Python dataclass in `phoneme_model.py`. Pass it to any function that accepts a `config` argument.

```python
from phoneme_model import PhonemeBoundaryConfig

config = PhonemeBoundaryConfig(
    device="cuda",
    use_fp16=True,
)
```

### Model

| Key          | Default                                  | Description                        |
|--------------|------------------------------------------|------------------------------------|
| `model_name` | `"facebook/wav2vec2-lv-60-espeak-cv-ft"` | HuggingFace checkpoint to load     |
| `sample_rate`| 16000                                    | Expected input sample rate (Hz)    |
| `device`     | `"cuda"` if available, else `"cpu"`      | Inference device                   |
| `batch_size` | 1                                        | Batch size (currently single-file) |
| `use_fp16`   | False                                    | Half-precision inference on GPU    |

### CTC Decoding

| Key                       | Default | Description                                     |
|---------------------------|---------|-------------------------------------------------|
| `collapse_repeated_tokens`| True    | Merge consecutive identical token frames        |
| `remove_blank_tokens`     | True    | Drop blank (ID = 0) frames from output          |
| `blank_token_id`          | 0       | CTC blank token index (always 0 in HF Wav2Vec2) |

`collapse_repeated_tokens=False` gives one segment per encoder frame — useful for debugging alignment.

### Post-processing A — long-segment splitting

| Key               | Default | Description                                                      |
|-------------------|---------|------------------------------------------------------------------|
| `max_segment_ms`  | 300.0   | Split CTC segments longer than this (ms). Set 0 to disable.     |
| `min_split_prob`  | 0.25    | Minimum alternative-token probability required to justify a split|

In practice this rarely triggers for singing (typical CTC segments are 20–120 ms).

### Post-processing B — blank-region phoneme recovery

Brief consonants in singing are often suppressed by CTC blank dominance: the blank token wins the argmax even when the consonant has an elevated probability in the logits. Enabling `blank_region_scan` recovers these missed boundaries.

| Key                 | Default | Description                                                              |
|---------------------|---------|--------------------------------------------------------------------------|
| `blank_region_scan` | True    | Scan blank gaps between segments for suppressed phonemes                 |
| `min_phoneme_prob`  | 0.05    | Minimum non-blank probability to insert a recovered segment              |
| `min_gap_frames`    | 2       | Skip gaps shorter than this many frames (2 frames = 40 ms; filters noise)|

These defaults were tuned on GTSinger singing data (Breathy + Glissando) and yield approximately +4pp F1 over raw CTC decoding at 30 ms boundary tolerance.

---

## Output Format

### Python dict

```python
{
    "phonemes":  ["h", "ɛ", "l", "oʊ", ...],         # IPA strings
    "boundaries": [[0.02, 0.08], [0.08, 0.16], ...],  # [start_s, end_s] per phoneme
    "segments":  [PhonemeSegment(...)],                 # full objects (when return_segments=True)
    "words":     [{"phonemes": [...], "start_time": ..., "end_time": ...}],  # when word_grouping=True
    "metadata": {
        "audio_file": "singing.wav",
        "duration_s": 7.32,
        "num_phonemes": 17,
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "device": "cuda"
    }
}
```

### PhonemeSegment fields

| Field         | Type  | Description                                   |
|---------------|-------|-----------------------------------------------|
| `phoneme`     | str   | IPA phoneme string                            |
| `start_time`  | float | Segment start in seconds                      |
| `end_time`    | float | Segment end in seconds                        |
| `confidence`  | float | Mean softmax probability of the winning token |
| `frame_start` | int   | First encoder frame (inclusive)               |
| `frame_end`   | int   | Last encoder frame (inclusive)                |

---

## Programmatic API Reference

### Load audio

```python
from phoneme_model import load_audio

audio, sr = load_audio("singing.wav", target_sr=16000)
# audio: 1-D float32 CPU tensor, shape [num_samples]
```

### Load model

```python
from phoneme_model import load_model, PhonemeBoundaryConfig

config = PhonemeBoundaryConfig(device="cuda")
model, processor = load_model(config)
# Call once; reuse model and processor across all files
```

### Run inference

```python
from phoneme_model import run_inference

logits, input_values = run_inference(audio, model, processor, config)
# logits: [1, T, vocab_size]
```

### Extract phoneme boundaries

```python
from phoneme_model import extract_phoneme_boundaries

phonemes, time_boundaries, frame_ranges, confidences = extract_phoneme_boundaries(
    logits, model, processor, config
)
# Four parallel lists — one entry per detected phoneme
```

### Create segments

```python
from phoneme_model import create_phoneme_segments

segments = create_phoneme_segments(phonemes, time_boundaries, frame_ranges, confidences)
for seg in segments:
    print(f"{seg.phoneme}  {seg.start_time:.3f}s – {seg.end_time:.3f}s  conf={seg.confidence:.3f}")
```

### All-in-one pipeline

```python
from phoneme_model import extract_phoneme_boundaries_from_audio, PhonemeBoundaryConfig

config = PhonemeBoundaryConfig(device="cuda")
result = extract_phoneme_boundaries_from_audio(
    "singing.wav",
    config=config,
    return_segments=True,
    word_grouping=True,
)
```

### Evaluate against reference

```python
from phoneme_model import compute_boundary_metrics

metrics = compute_boundary_metrics(predicted_segments, reference_segments, tolerance_ms=50.0)
print(f"F1:  {metrics['f1']:.4f}")
print(f"MAE: {metrics['mae_ms']:.2f} ms")
```

### Visualize

```python
from phoneme_model import plot_phoneme_boundaries

plot_phoneme_boundaries(audio, result["segments"], sample_rate=16000, save_path="plot.png")
```

---

## Model Architecture

```
Input: Raw Waveform  [num_samples]  (16 kHz mono float32)
           │
   ┌───────▼────────────────────────────┐
   │  Conv1D Feature Encoder ×7         │  strides: [5,2,2,2,2,2,2] → total stride 320
   │                                    │  output: ~50 frames/s  (20 ms per frame)
   └───────┬────────────────────────────┘
           │  [T, 512]
   ┌───────▼────────────────────────────┐
   │  Transformer Context Network ×24   │  self-attention over full sequence
   └───────┬────────────────────────────┘
           │  [T, 1024]
   ┌───────▼────────────────────────────┐
   │  CTC Linear Head                   │  Linear(1024 → vocab_size ~400)
   └───────┬────────────────────────────┘
           │  logits [T, vocab_size]
           │  greedy argmax → [T]
           │  collapse repeats + remove blanks
           ▼
   phoneme segments with timing
```

Frame rate: `total_conv_stride / sample_rate = 320 / 16000 = 20 ms per frame`

Checkpoint: `facebook/wav2vec2-lv-60-espeak-cv-ft` — trained on Common Voice, multilingual, IPA phoneme set via eSpeak NG (~400 tokens).

---

## Tips & Tuning

**If all phonemes decode as `<unk>`:**
- Root cause: `model.config.id2label` is absent or incomplete for this checkpoint, so the IPA tokens (stored in `added_tokens.json`) are never loaded into the lookup table
- Fix applied in `build_id2phoneme`: now reads directly from `processor.tokenizer._added_tokens_decoder`, which is populated by `Wav2Vec2PhonemeCTCTokenizer.from_pretrained()` — the only reliable source for IPA tokens
- Do not replace `load_model()` with `AutoProcessor.from_pretrained()`; that bypasses `Wav2Vec2PhonemeCTCTokenizer` entirely

**If precision is low (too many spurious boundaries):**
- The model over-segments on long sustained notes — this is expected for singing vs speech
- Use `group_by_words=True` to aggregate fine segments into word-level boundaries

**If recall is low (missing boundaries):**
- Singing vibrato and tempo variation shift boundaries outside the 50 ms window
- Widen the tolerance: `compute_boundary_metrics(..., tolerance_ms=80.0)`

**If inference is slow on CPU:**
- Set `use_fp16=True` if your CPU supports it (minor speedup)
- Load the model once with `load_model()` and reuse across files — don't reload per file

**If CUDA is out of memory:**
- Set `device="cpu"` in `PhonemeBoundaryConfig`
- Or enable `use_fp16=True` to halve VRAM from ~1 GB to ~500 MB

**If audio fails to load:**
- `torchaudio` falls back to `soundfile` automatically when FFmpeg is missing on Windows
- Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`

---

## File Responsibilities

| File               | What to change it for                                        |
|--------------------|--------------------------------------------------------------|
| `phoneme_model.py` | Config dataclass, pipeline logic, CTC alignment, CLI        |
| `evaluate.py`      | Batch evaluation strategy, aggregate statistics, CSV output  |
| `examples.py`      | Usage examples and integration patterns                      |
| `test.py`          | Unit tests — run with `pytest test.py -v`                    |
| `requirements.txt` | Dependency versions                                          |
