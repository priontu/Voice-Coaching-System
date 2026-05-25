# Phoneme Boundary Detection Module

A production-grade Python pipeline for extracting precise phoneme boundaries from singing voice audio using **Wav2Vec2Phoneme** with **CTC alignment**.

---

## Table of Contents

1. [Overview](#overview)
2. [Model](#model)
3. [Architecture & Pipeline](#architecture--pipeline)
4. [Implementation](#implementation)
   - [Configuration](#configuration)
   - [Audio Loading](#audio-loading)
   - [Model Loading](#model-loading)
   - [Vocabulary Mapping](#vocabulary-mapping)
   - [Inference](#inference)
   - [CTC Alignment](#ctc-alignment)
   - [Boundary Extraction](#boundary-extraction)
   - [Segment Construction](#segment-construction)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Visualization](#visualization)
5. [Usage](#usage)
6. [CLI](#cli)
7. [Evaluation Results](#evaluation-results)
8. [Output Format](#output-format)
9. [Installation](#installation)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This module takes a raw audio file and outputs time-stamped IPA phoneme segments — start time, end time, and confidence per phoneme. It is designed for singing voice analysis where precise boundary timing matters more than text transcription accuracy.

**Pipeline summary:**

```
Audio File
    │
    ▼
load_audio()           — resample to 16 kHz, mono, normalise
    │
    ▼
load_model()           — Wav2Vec2PhonemeCTCTokenizer + Wav2Vec2Processor
    │
    ▼
run_inference()        — processor → input_values → model → logits [1, T, V]
    │
    ▼
extract_phoneme_boundaries()
    ├── ctc_align()    — argmax → collapse repeats → remove blanks
    ├── build_id2phoneme() — token ID → IPA string
    └── compute_frame_duration() — conv_stride product / sample_rate
    │
    ▼
create_phoneme_segments()  — PhonemeSegment list with timing + confidence
    │
    ├── group_by_words()   — optional word-level grouping
    ├── plot_phoneme_boundaries() — optional waveform visualisation
    └── compute_boundary_metrics() — evaluation vs TextGrid ground truth
```

---

## Model

| Property | Value |
|---|---|
| Checkpoint | `facebook/wav2vec2-lv-60-espeak-cv-ft` |
| Architecture | Wav2Vec2 encoder + CTC linear head |
| Paper | *Simple and Effective Zero-shot Cross-lingual Phoneme Recognition* (Xu et al., 2021) |
| Training data | Common Voice, multilingual |
| Phoneme set | IPA (eSpeak NG), ~400 tokens including added_tokens |
| Input | 16 kHz mono float32 audio |
| Output | Frame-level logits, shape `[batch, time_frames, vocab_size]` |
| Frame duration | ~20 ms per frame (derived from `conv_stride` product ÷ sample rate) |
| Tokenizer class | `Wav2Vec2PhonemeCTCTokenizer` |
| Processor class | `Wav2Vec2Processor` |

The model uses `Wav2Vec2PhonemeCTCTokenizer` — **not** `AutoProcessor` — because this tokenizer correctly loads IPA phoneme tokens from `added_tokens.json`. Using `AutoProcessor` alone causes all phoneme labels to decode as `<unk>`.

---

## Architecture & Pipeline

### Wav2Vec2 Encoder

Wav2Vec2 processes raw waveform through a stack of strided 1-D convolutions (the feature encoder) followed by a Transformer context network. The convolutional feature encoder downsamples audio at a fixed stride, producing one encoder frame roughly every 20 ms.

```
Raw waveform  →  [Conv1D × 7 with strides]  →  [Transformer × 24]  →  Logits
16000 samples/s                  ~50 frames/s                 [T, vocab_size]
```

### CTC Decoding

Connectionist Temporal Classification (CTC) allows the model to be trained without frame-level alignment. At inference:

1. **Greedy argmax** — for every frame, take the token ID with the highest logit
2. **Collapse repeats** — consecutive identical tokens merge into one segment
3. **Remove blanks** — blank token (ID = 0) frames are dropped
4. **Time conversion** — frame index × frame_duration → seconds

```
Frame IDs:   [0, 1, 1, 0, 2, 2, 0, 3, 0]
              blank  ─────  blank  ─────  blank  single  blank
              │      token 1       token 2       token 3
Collapsed:   [     1,            2,            3     ]
Times (20ms): 0.02–0.06s        0.08–0.14s    0.14–0.16s
```

---

## Implementation

### Configuration

```python
@dataclass
class PhonemeBoundaryConfig:
    model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    sample_rate: int = 16000
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch_size: int = 1
    use_fp16: bool = False
    collapse_repeated_tokens: bool = True   # standard CTC — merge consecutive same-token frames
    remove_blank_tokens: bool = True         # drop blank (ID=0) frames from output
    blank_token_id: int = 0                  # CTC blank is always index 0 in HF Wav2Vec2

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
```

`device` accepts either a string `"cuda"` / `"cpu"` or a `torch.device` object. The `__post_init__` normalises it so downstream code always sees `torch.device`.

---

### Audio Loading

```python
def load_audio(filepath: str, target_sr: int = 16000, mono: bool = True) -> Tuple[torch.Tensor, int]:
```

- Primary loader: `torchaudio.load()`
- Fallback: `soundfile.read()` — used when torchaudio fails (missing FFmpeg on Windows)
- Multi-channel audio is mixed to mono by averaging channels
- Resampled to `target_sr` using `torchaudio.transforms.Resample`
- Peak-normalised: divided by `abs(waveform).max()`
- Returns a **1-D CPU float32 tensor** — shape `[num_samples]`

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`

---

### Model Loading

```python
def load_model(config: PhonemeBoundaryConfig) -> Tuple[AutoModelForCTC, Wav2Vec2Processor]:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name)
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(config.model_name)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = AutoModelForCTC.from_pretrained(config.model_name)
    model = model.to(config.device).eval()
    return model, processor
```

**Why construct the processor manually?**

`AutoProcessor.from_pretrained()` loads a generic tokenizer that reads the base vocabulary only. The IPA phoneme tokens for this model are stored in a separate `added_tokens.json` file. `Wav2Vec2PhonemeCTCTokenizer` reads both files, giving the correct full vocabulary. Constructing `Wav2Vec2Processor` from the explicit feature extractor and tokenizer is the correct approach for Wav2Vec2Phoneme models.

The model is moved to `config.device` and set to `eval()` mode (disables dropout) before returning.

---

### Vocabulary Mapping

```python
def build_id2phoneme(model: AutoModelForCTC, processor: Wav2Vec2Processor) -> Dict[int, str]:
```

Builds a `{token_id: phoneme_string}` dictionary. Two-level priority:

1. **`model.config.id2label`** — present on most CTC fine-tunes, directly maps ID → IPA string
2. **`processor.tokenizer.convert_ids_to_tokens(range(vocab_size))`** — fallback; correctly resolves added IPA tokens that `get_vocab()` misses

```python
def compute_frame_duration(model: AutoModelForCTC, sample_rate: int) -> float:
```

Derives the seconds-per-frame value from the model's convolutional stride configuration:

```python
total_stride = product(model.config.conv_stride)  # e.g. [5,2,2,2,2,2,2] → 320
frame_duration = total_stride / sample_rate         # 320 / 16000 = 0.02s = 20ms
```

Falls back to `0.02` (20 ms) if `conv_stride` is absent from the model config.

---

### Inference

```python
def run_inference(audio, model, processor, config) -> Tuple[torch.Tensor, torch.Tensor]:
    audio_np = audio.cpu().numpy()
    with torch.no_grad():
        inputs = processor(audio_np, sampling_rate=config.sample_rate,
                           return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(config.device)
        logits = model(input_values).logits   # [1, T, vocab_size]
    return logits, input_values
```

- `processor()` applies mean/variance normalisation (z-score) to the raw waveform and returns `input_values`
- `torch.no_grad()` disables gradient computation — required for inference; halves memory use
- Logits shape: `[1, T, V]` where T = number of encoder frames, V = vocab size (~400)

---

### CTC Alignment

```python
def ctc_align(
    pred_ids: np.ndarray,
    collapse_repeated: bool = True,
    remove_blanks: bool = True,
    blank_id: int = 0,
) -> Tuple[List[int], List[Tuple[int, int]]]:
```

Converts the frame-level argmax sequence into a list of unique tokens with their frame ranges.

```python
for i, token_id in enumerate(pred_ids):
    if remove_blanks and token_id == blank_id:
        continue                            # skip blank frames

    if aligned_tokens and token_id == aligned_tokens[-1]:
        if collapse_repeated:
            frame_ranges[-1][1] = i         # extend the current segment's end frame
            continue
        # collapse_repeated=False: each frame is its own segment
        frame_ranges.append([i, i])
        aligned_tokens.append(token_id)
    else:
        frame_ranges.append([i, i])         # new token — start a fresh segment
        aligned_tokens.append(token_id)
```

Returns `(aligned_tokens, frame_ranges)` where each `frame_range` is an **inclusive** `[start, end]` pair.

**Key correctness fix:** when `collapse_repeated=False`, the original code extended `frame_ranges[-1][1]` regardless, creating overlapping ranges. The fix gates that mutation inside `if collapse_repeated:` only.

---

### Boundary Extraction

```python
def extract_phoneme_boundaries(
    logits, model, processor, config, input_length=None
) -> Tuple[List[str], List[Tuple[float,float]], List[Tuple[int,int]], List[float]]:
```

Orchestrates decoding from logits to timed phoneme data:

```python
# 1. Greedy argmax — use [0] not .squeeze() to avoid scalar collapse on T=1
pred_ids = torch.argmax(logits_cpu, dim=-1)[0].numpy()   # [T]

# 2. CTC alignment
aligned_tokens, frame_ranges = ctc_align(pred_ids, ...)

# 3. Token ID → phoneme string (with <unk> fallback for unknown IDs)
id2phoneme = build_id2phoneme(model, processor)
phonemes = [id2phoneme.get(tid, "<unk>") for tid in aligned_tokens]

# 4. Per-segment confidence: mean softmax probability of the winning token
probs = torch.softmax(logits_cpu[0], dim=-1).numpy()   # [T, V]
confidences = [probs[fs:fe+1, tid].mean() for (fs,fe), tid in zip(frame_ranges, aligned_tokens)]

# 5. Frame index → seconds
frame_dur = compute_frame_duration(model, config.sample_rate)
time_boundaries = [(fs * frame_dur, (fe + 1) * frame_dur) for fs, fe in frame_ranges]
```

Returns four parallel lists: `phonemes`, `time_boundaries`, `frame_ranges`, `confidences`.

---

### Segment Construction

```python
@dataclass
class PhonemeSegment:
    phoneme: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    frame_start: int = 0
    frame_end: int = 0
```

`create_phoneme_segments()` zips the four parallel lists into `PhonemeSegment` objects:

```python
PhonemeSegment(
    phoneme=ph,
    start_time=round(t0, 4),
    end_time=round(t1, 4),
    confidence=round(confidences[i], 4),
    frame_start=frame_ranges[i][0],
    frame_end=frame_ranges[i][1],
)
```

`group_by_words()` splits segments on the `|` word-boundary token (used by the eSpeak vocabulary) and groups the phonemes in between into word dictionaries with aggregate timing.

---

### Evaluation Metrics

```python
def compute_boundary_metrics(
    predicted: List[PhonemeSegment],
    reference: List[PhonemeSegment],
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
```

Evaluates **boundary timing accuracy** — does not require matching phoneme labels. A predicted boundary matches if it falls within `tolerance_ms` of any reference boundary.

Boundaries collected: all segment start times + the final segment's end time.

```python
pred_b = [s.start_time for s in predicted] + [predicted[-1].end_time]
ref_b  = [s.start_time for s in reference] + [reference[-1].end_time]

for pt in pred_b:
    nearest = min(abs(pt - rt) for rt in ref_b)
    if nearest <= tolerance:
        matches += 1
        total_error += nearest

precision = matches / len(pred_b)
recall    = matches / len(ref_b)
f1        = 2 * precision * recall / (precision + recall)
mae_ms    = (total_error / matches) * 1000
```

| Metric | Definition |
|---|---|
| Precision | Fraction of predicted boundaries that match a reference boundary |
| Recall | Fraction of reference boundaries that are covered by a prediction |
| F1 | Harmonic mean of precision and recall |
| MAE (ms) | Mean absolute timing error of matched boundary pairs |

---

### Visualization

`plot_phoneme_boundaries()` renders a two-panel matplotlib figure:
- **Top panel**: waveform amplitude over time
- **Bottom panel**: colour-coded phoneme segments with labels at segment midpoints

Saves to PNG when `save_path` is given, otherwise calls `plt.show()`.

---

## Usage

### Basic

```python
from phoneme_model import extract_phoneme_boundaries_from_audio

result = extract_phoneme_boundaries_from_audio("audio.wav")

for seg in result["segments"][:5]:
    print(f"{seg.phoneme}  {seg.start_time:.3f}s – {seg.end_time:.3f}s  conf={seg.confidence:.3f}")
```

### With Configuration

```python
from phoneme_model import extract_phoneme_boundaries_from_audio, PhonemeBoundaryConfig

config = PhonemeBoundaryConfig(
    model_name="facebook/wav2vec2-lv-60-espeak-cv-ft",
    device="cuda",
    collapse_repeated_tokens=True,
    remove_blank_tokens=True,
)

result = extract_phoneme_boundaries_from_audio(
    "singing.wav",
    config=config,
    return_segments=True,
    word_grouping=True,
)
```

### Visualization

```python
from phoneme_model import load_audio, plot_phoneme_boundaries

audio, sr = load_audio("singing.wav")
plot_phoneme_boundaries(audio, result["segments"], sample_rate=sr, save_path="plot.png")
```

### Evaluation Against TextGrid Ground Truth

```python
import textgrid
from phoneme_model import extract_phoneme_boundaries_from_audio, compute_boundary_metrics, PhonemeSegment

tg = textgrid.TextGrid.fromFile("reference.TextGrid")
tier = tg.getFirst("phone")

ref_segments = [
    PhonemeSegment(phoneme=iv.mark, start_time=iv.minTime, end_time=iv.maxTime)
    for iv in tier
    if iv.mark and iv.mark not in ("sil", "pau", "", "<SP>")
]

result = extract_phoneme_boundaries_from_audio("audio.wav")
metrics = compute_boundary_metrics(result["segments"], ref_segments, tolerance_ms=50.0)

print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall:    {metrics['recall']:.2%}")
print(f"F1:        {metrics['f1']:.4f}")
print(f"MAE:       {metrics['mae_ms']:.2f} ms")
```

### Batch Evaluation (evaluate.py)

```bash
py evaluate.py --gtsinger-dir ./gtsinger/English --device cuda --max-files 10 --verbose
```

`evaluate.py` discovers all `.TextGrid` / `.wav` pairs under the given directory, runs the pipeline on each, and saves aggregate results to JSON. Options:

| Flag | Default | Description |
|---|---|---|
| `--gtsinger-dir` | required | Root of GTSinger dataset |
| `--max-files` | all | Limit files processed |
| `--tolerance-ms` | 50.0 | Boundary match window |
| `--device` | cuda | `cuda` or `cpu` |
| `--output` | `gtsinger_evaluation_results.json` | Output JSON path |
| `--analyze` | off | Generate CSV + matplotlib plots |
| `--verbose` | off | Per-file metric printout |

---

## CLI

```bash
# Basic extraction
python phoneme_model.py audio.wav

# Save to JSON
python phoneme_model.py audio.wav --output results.json

# Visualise boundaries
python phoneme_model.py audio.wav --output results.json --plot

# Word-level grouping
python phoneme_model.py audio.wav --words

# Evaluate against a reference JSON
python phoneme_model.py audio.wav --reference ground_truth.json

# All options
python phoneme_model.py singing.wav --output results.json --plot --words --device cuda
```

---

## Evaluation Results

Evaluated on 10 English files from the **GTSinger** singing voice dataset using a 50 ms tolerance window.

### Per-File Results

| File | Duration | Predictions | References | Precision | Recall | F1 | MAE (ms) |
|---|---|---|---|---|---|---|---|
| 0000.wav | 7.32 s | 17 | 22 | 0.778 | 0.609 | 0.683 | 29.1 |
| 0001.wav | 10.99 s | 35 | 41 | 0.639 | 0.548 | 0.590 | 28.2 |
| 0002.wav | 7.09 s | 22 | 20 | 0.609 | 0.667 | 0.636 | 26.1 |
| 0003.wav | 6.74 s | 17 | 18 | 0.778 | 0.737 | **0.757** | 24.3 |
| 0004.wav | 10.37 s | 36 | 40 | 0.784 | 0.707 | 0.744 | 29.7 |
| 0005.wav | 6.03 s | 15 | 23 | 0.750 | 0.500 | 0.600 | 28.0 |
| 0006.wav | 7.76 s | 17 | 22 | 0.667 | 0.522 | 0.585 | 27.8 |
| 0007.wav | 6.67 s | 18 | 23 | 0.789 | 0.625 | 0.698 | 29.5 |
| 0008.wav | 8.17 s | 26 | 34 | 0.667 | 0.514 | **0.581** | 26.9 |
| 0009.wav | 5.96 s | 12 | 13 | 0.615 | 0.571 | 0.593 | 17.6 |

### Aggregate

| Metric | Value |
|---|---|
| Avg Precision | **0.7075** |
| Avg Recall | **0.6000** |
| Avg F1 | **0.6466** |
| Median F1 | 0.6182 |
| Std F1 | 0.0648 |
| Min F1 | 0.5806 |
| Max F1 | 0.7568 |
| Avg MAE | **26.73 ms** |
| Success rate | 10 / 10 |

### Interpretation

- **MAE ≈ 26.7 ms** is approximately one encoder frame (20 ms), indicating timing quality is near the model's resolution limit — no systematic drift.
- **Precision (0.71) > Recall (0.60)** consistently — the model under-predicts boundaries rather than over-segmenting. Roughly 75 % of reference boundaries are found.
- **F1 0.58 – 0.76** is moderate and expected: the model was trained on read speech, not singing. Singing involves sustained notes, vibrato, and tempo variation that differ significantly from training distribution.
- Boundary timing quality is independent of phoneme label accuracy — the evaluation compares timing only.

---

## Output Format

### Python Dict

```python
{
    "phonemes": ["h", "ɛ", "l", "oʊ", ...],          # IPA strings
    "boundaries": [[0.02, 0.08], [0.08, 0.16], ...],  # [start_s, end_s] per phoneme
    "segments": [PhonemeSegment(...)],                  # full objects (when return_segments=True)
    "words": [{"phonemes": [...], "start_time": ..., "end_time": ...}],  # when word_grouping=True
    "metadata": {
        "audio_file": "singing.wav",
        "duration_s": 7.32,
        "num_phonemes": 17,
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "device": "cuda"
    }
}
```

### JSON (--output flag)

```json
{
    "phonemes": ["h", "ɛ", "l", "oʊ"],
    "boundaries": [[0.02, 0.08], [0.08, 0.16]],
    "segments": [
        {
            "phoneme": "h",
            "start_time": 0.02,
            "end_time": 0.08,
            "confidence": 0.9423,
            "frame_start": 1,
            "frame_end": 4
        }
    ],
    "metadata": {
        "audio_file": "singing.wav",
        "duration_s": 7.32,
        "num_phonemes": 17,
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "device": "cuda"
    }
}
```

---

## Installation

### Dependencies

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers soundfile numpy matplotlib pandas textgrid
```

### eSpeak NG (Windows)

`Wav2Vec2PhonemeCTCTokenizer` requires eSpeak NG on Windows. The module sets these environment variables automatically:

```python
os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", r"C:\Program Files\eSpeak NG\espeak-ng.exe")
os.environ.setdefault("ESPEAKNG_PATH",           r"C:\Program Files\eSpeak NG")
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
```

Download: https://github.com/espeak-ng/espeak-ng/releases

### Pre-download Model (~360 MB)

```python
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2PhonemeCTCTokenizer, AutoModelForCTC

Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| All phonemes are `<unk>` | Wrong tokenizer class; `get_vocab()` misses IPA tokens in `added_tokens.json` | Use `Wav2Vec2PhonemeCTCTokenizer` + `convert_ids_to_tokens()` |
| `AttributeError: 'Wav2Vec2Processor' has no attribute 'id2label'` | Called `processor.id2label` instead of `model.config.id2label` | Use `build_id2phoneme(model, processor)` |
| `UnicodeEncodeError` on Windows | Unicode status symbols (✓, ✗) can't be encoded in CP1252 console | Use ASCII alternatives `[OK]`, `[+++]`, `[---]` |
| `torch.device` not JSON serialisable | `torch.device` object written directly to JSON metadata | Use `str(config.device)` in output dict |
| CUDA out of memory | Model too large for VRAM | Set `device="cpu"` or enable `use_fp16=True` |
| Audio fails to load | FFmpeg missing on Windows | torchaudio falls back to soundfile automatically |
| Model reloaded per file | `load_model` called inside the per-file loop | Load once and pass `model`/`processor` to each file |

---

## Memory & Speed

| Component | Memory |
|---|---|
| Model weights (FP32) | ~360 MB |
| Inference (batch=1, 10 s audio) | ~500 MB GPU |
| **Total VRAM** | **~1 GB** |

| Device | 10 s audio | Speed |
|---|---|---|
| GPU (RTX 30xx) | ~1.2 s | ~8× realtime |
| CPU (8-core) | ~25 s | ~0.4× realtime |

---

## Reference

**Model paper**: Xu et al. (2021). *Simple and Effective Zero-shot Cross-lingual Phoneme Recognition*. https://arxiv.org/abs/2109.11680

**Wav2Vec2 original**: Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS 2020.

**HuggingFace model doc**: https://huggingface.co/docs/transformers/model_doc/wav2vec2_phoneme

**Checkpoint**: https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft

**GTSinger dataset**: used for boundary evaluation (English subset, 10 files, TextGrid annotations)
