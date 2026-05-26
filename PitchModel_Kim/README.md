# Pitch Estimator + VAD Pipeline

VAD-gated torchcrepe pitch extraction for singing voice analysis.  
Part of the VocalCoach interpretable singing evaluation system.

---

## What this does

```
WAV audio
    → load + resample to 16 kHz
    → WebRTC VAD  →  voiced/unvoiced mask  [T_vad]
    → torchcrepe  →  raw F0 curve          [T_pitch]
    → align VAD mask to pitch frame rate
    → intersect + smooth + gap-fill
    → pitch_data.json: [{time, f0, voiced}, ...]
    → pitch_score.py: PitchAcc50, MACE, PitchRMSE, NotePitchAcc50
```

The pipeline does **not** transcribe notes or decode lyrics.  
It produces only **frame-level F0** and a **voiced/unvoiced mask** for downstream scoring.

---

## Project structure

```
Pitch Model/
├── utils.py             ← audio I/O, PCM conversion, JSON serialization, device detection
├── vad.py               ← WebRTC VAD wrapper + energy-based fallback
├── alignment.py         ← resamples VAD mask from VAD frame rate to pitch frame rate
├── pitch_wrapper.py     ← torchcrepe / pYIN / custom-callable wrapper
├── fusion.py            ← VAD–pitch intersection, gap fill, median filter, smoothing
├── inference.py         ← PitchVADPipeline class + CLI entrypoint
├── visualization.py     ← waveform, VAD overlay, pitch contour plots
├── metrics.py           ← MACE, PitchRMSE, PitchAccN, voiced/unvoiced accuracy
│
├── pitch_score.py       ← (existing) frame + note-level scoring against MusicXML
└── test_pitch.py        ← (existing) standalone torchcrepe extraction
```

---

## Dependencies

### Python version

Python 3.10 or later is required.

### Install

```bash
pip install torch torchcrepe librosa soundfile scipy numpy matplotlib music21
```

Full package list:

| Package           | Purpose                                          |
|-------------------|--------------------------------------------------|
| torch             | model inference, GPU support                     |
| torchcrepe        | CREPE pitch estimation (PyTorch port)            |
| librosa           | audio resampling, pYIN fallback backend          |
| soundfile         | broad audio format support (WAV, FLAC, OGG)      |
| scipy             | median filter, Gaussian smoothing, interpolation |
| numpy             | array ops                                        |
| matplotlib        | visualization                                    |
| music21           | MusicXML parsing (used by pitch_score.py)        |
| webrtcvad-wheels  | WebRTC VAD — see VAD install section below       |

> **GPU** — the pipeline auto-detects CUDA or Apple MPS. No extra flags needed.  
> Install the CUDA build of PyTorch from https://pytorch.org/get-started if you want GPU inference.

### Voice Activity Detection

Both `webrtcvad-wheels` and `webrtcvad` compile a C extension at install time and
**require Microsoft C++ Build Tools on Windows**. Neither has pre-built wheels for
Python 3.14 yet.

**Option A — Install the free MSVC build tools, then install webrtcvad:**

1. Download and install **Microsoft C++ Build Tools** from  
   https://visualstudio.microsoft.com/visual-cpp-build-tools/  
   (select the "Desktop development with C++" workload, ~5 GB)
2. Restart your terminal, then:

```bash
pip install webrtcvad-wheels
```

**Option B — Use Python 3.11 or 3.12** (pre-built wheels exist for those versions):

```bash
pip install webrtcvad-wheels   # works out of the box on Python 3.11 / 3.12
```

**Option C — Skip VAD installation entirely (recommended for now):**

The pipeline has a built-in energy-based VAD fallback that activates automatically
when webrtcvad is not installed. No code changes needed — just run `inference.py`
and the fallback is used silently. Results are slightly noisier on recordings with
heavy background noise, but work well for typical singing audio.

---

## Quick Start

### Step 1 — Extract pitch

Run the pipeline on a WAV file. GPU is used automatically if available.

```bash
cd "c:\Users\kimhu\Documents\VSCode\MusicAI\VocalCoach\Pitch Model"

python inference.py --audio sample.wav
```

This writes `pitch_data.json` (compatible with `pitch_score.py`) and prints a summary:

```
──────────────────────────────────────────────────
  Frames:        3120
  Voiced frames: 2184
  Voiced ratio:  70.0%
  Voiced dur:    21.84s
  Output JSON:   pitch_data.json
──────────────────────────────────────────────────

  F0 min:    168.4 Hz
  F0 max:    587.3 Hz
  F0 median: 311.1 Hz
  F0 mean:   298.6 Hz
```

**Optional flags:**

```bash
# Save a combined waveform / VAD / pitch plot
python inference.py --audio sample.wav --visualize

# Force CPU (e.g. for reproducibility)
python inference.py --audio sample.wav --device cpu

# Tune VAD sensitivity (0 = least aggressive, 3 = most)
python inference.py --audio sample.wav --vad-mode 1

# Use pYIN instead of torchcrepe
python inference.py --audio sample.wav --backend pyin

# Skip VAD entirely (use torchcrepe's own voiced detection)
python inference.py --audio sample.wav --no-vad
```

---

### Step 2 — Score against MusicXML

`pitch_score.py` reads `pitch_data.json` and `test.musicxml` exactly as before.  
No changes to the existing scorer are needed.

```bash
python pitch_score.py
```

Console output:

```
Frame-Level Pitch Scoring Results
---------------------------------
Pitch Accuracy ±50 cents: 84.23%
Mean Absolute Cent Error: 22.41 cents
Pitch RMSE: 31.07 cents
Compared voiced frames: 2184

Worst Frame-Level Pitch Mismatches
----------------------------------
Mismatch #1
Time: 8.34 sec
Expected: 329.63 Hz / MIDI 64.00
Detected: 311.13 Hz / MIDI 63.00
Error: -98.07 cents (flat)

Note-Level Pitch Correctness
----------------------------
NotePitchAcc50: 78.57%
Correct notes: 11 / 14
Total reference notes: 14
```

Saved output files:

```
pitch_vs_reference.png         ← predicted vs reference F0 overlay
cent_error_over_time.png       ← per-frame cent error curve
pitch_accuracy_summary.png     ← bar chart of PitchAcc50 + NotePitchAcc50
note_level_pitch_errors.png    ← per-note median cent error
note_pitch_results.json        ← full note-level results
```

---

### Step 3 — Visualize the VAD + pitch output

```bash
python inference.py --audio sample.wav --visualize
```

Saves `pitch_data_vad_pitch.png` — a three-panel figure:

| Panel  | Content                                        |
|--------|------------------------------------------------|
| Top    | Blue waveform with green voiced-region shading |
| Middle | Binary VAD mask over time                      |
| Bottom | Red pitch contour dots (+ orange reference F0 if provided) |

---

### Step 4 — Evaluate pipeline metrics

Use `metrics.py` directly to measure pitch and VAD accuracy against a reference:

```python
from metrics import evaluate_all, print_metrics_report
from utils import load_pitch_json
import numpy as np

times, f0, voiced = load_pitch_json("pitch_data.json")

# reference_f0 must be aligned to the same timestamps
results = evaluate_all(
    predicted_f0=f0,
    reference_f0=reference_f0,
    predicted_voiced=voiced,
)
print_metrics_report(results)
```

Console output:

```
Evaluation Metrics
========================================
  Voiced frames compared  : 2184
  MACE                    : 22.41 cents
  PitchRMSE               : 31.07 cents
  PitchAcc25              : 61.20%
  PitchAcc50              : 84.23%
  PitchAcc100             : 96.45%
  OctaveErrorRate         : 0.32%
========================================
```

---

## CLI Reference

```
python inference.py [OPTIONS]

Required:
  --audio, -a PATH              Input WAV file

Output:
  --output, -o PATH             Output JSON (default: pitch_data.json)
  --visualize                   Save a combined waveform / VAD / pitch plot

Pitch model:
  --backend {torchcrepe,pyin}   Pitch backend (default: torchcrepe)
  --device DEVICE               auto (default), cpu, cuda, cuda:0, mps
  --hop-length N                Hop size in samples (default: 160 = 10 ms @ 16 kHz)
  --fmin HZ                     Minimum F0 in Hz (default: 50)
  --fmax HZ                     Maximum F0 in Hz (default: 1000)
  --periodicity-threshold FLOAT torchcrepe confidence cutoff (default: 0.21)

VAD:
  --vad-mode {0,1,2,3}          WebRTC aggressiveness (default: 2)
  --vad-frame-ms {10,20,30}     VAD frame duration in ms (default: 20)
  --no-vad                      Skip VAD — use torchcrepe's voiced detection only

Post-processing:
  --gap-fill N                  Max unvoiced frames to bridge by interpolation (default: 10)

Logging:
  --verbose, -v                 Enable DEBUG-level logging
```

---

## Configuration Reference

Configuration is passed as Python dataclass instances. The defaults below match `test_pitch.py` exactly so existing outputs are preserved.

### VAD (`VADConfig`)

| Field               | Default | Description                                                    |
|---------------------|---------|----------------------------------------------------------------|
| `aggressiveness`    | `2`     | WebRTC mode 0–3. Lower = keeps more speech; higher = more filtering |
| `frame_duration_ms` | `20`    | VAD frame length in ms. Must be 10, 20, or 30                 |
| `sample_rate`       | `16000` | Must match the loaded audio sample rate                        |
| `smoothing_window`  | `5`     | Median filter window over VAD frames to remove isolated flips  |
| `energy_threshold_db` | `-40` | Threshold for energy-based fallback VAD (dBFS)               |

### Pitch (`PitchConfig`)

| Field                    | Default      | Description                                               |
|--------------------------|--------------|-----------------------------------------------------------|
| `backend`                | `torchcrepe` | `"torchcrepe"`, `"pyin"`, or `"custom"`                   |
| `hop_length`             | `160`        | Hop size in samples — 160 @ 16 kHz = 10 ms per frame      |
| `fmin`                   | `50.0`       | Minimum detectable F0 in Hz                               |
| `fmax`                   | `1000.0`     | Maximum detectable F0 in Hz                               |
| `model_capacity`         | `"full"`     | torchcrepe model size: tiny / small / medium / large / full |
| `use_viterbi`            | `True`       | Viterbi decoding — smoother, better for sustained notes   |
| `periodicity_threshold`  | `0.21`       | Frames below this periodicity score → unvoiced            |
| `silence_threshold_db`   | `-60.0`      | Silence gate applied before periodicity thresholding      |
| `device`                 | `"auto"`     | `"auto"` selects CUDA → MPS → CPU                        |
| `batch_size`             | `1024`       | Batch size for GPU inference                              |

### Fusion (`FusionConfig`)

| Field                  | Default | Description                                                      |
|------------------------|---------|------------------------------------------------------------------|
| `unvoiced_fill`        | `"zero"`| Unvoiced frames → `0.0` (compatible with pitch_score.py) or `"nan"` |
| `median_filter_size`   | `5`     | Median kernel width for removing pitch outliers                  |
| `max_gap_fill_frames`  | `10`    | Max consecutive unvoiced frames to bridge by interpolation       |
| `smoothing_sigma`      | `1.0`   | Gaussian smoothing std in frames applied after median filter     |

---

## Output JSON format

`pitch_data.json` is a strict superset of the format `pitch_score.py` already expects.  
The extra `"voiced"` and `"midi"` fields are silently ignored by `load_pitch_data()`.

```json
{
  "audio_path": "sample.wav",
  "sample_rate": 16000,
  "hop_length": 160,
  "num_frames": 3120,
  "frames": [
    {"time": 0.000, "f0": 0.0,   "voiced": false, "midi": null},
    {"time": 0.010, "f0": 220.5, "voiced": true,  "midi": 57.0},
    {"time": 0.020, "f0": 221.3, "voiced": true,  "midi": 57.1}
  ]
}
```

---

## Programmatic API Reference

### Load and run the full pipeline

```python
from inference import PitchVADPipeline, PipelineConfig

pipeline = PitchVADPipeline()
result = pipeline.run("sample.wav")

times, f0, voiced = result.to_numpy()
print(f"Voiced ratio:    {result.voiced_ratio():.1%}")
print(f"Voiced duration: {result.voiced_duration():.2f}s")
```

### Configure VAD

```python
from inference import PitchVADPipeline, PipelineConfig
from vad import VADConfig

config = PipelineConfig(
    vad=VADConfig(aggressiveness=1, frame_duration_ms=20),
)
result = PitchVADPipeline(config).run("sample.wav")
```

### Configure pitch model

```python
from inference import PitchVADPipeline, PipelineConfig
from pitch_wrapper import PitchConfig

config = PipelineConfig(
    pitch=PitchConfig(fmin=100.0, fmax=800.0, periodicity_threshold=0.3),
)
result = PitchVADPipeline(config).run("sample.wav")
```

### Plug in a custom pitch model

```python
from pitch_wrapper import PitchModelWrapper

def my_model(audio, sr):
    # your model here
    return timestamps, f0_hz   # or (timestamps, f0_hz, confidence)

wrapper = PitchModelWrapper.from_callable(my_model)
times, f0, conf = wrapper.predict(audio, sr=16000)
```

### VAD only

```python
from utils import load_audio
from vad import run_vad

audio, sr = load_audio("sample.wav")
voiced_mask, timestamps = run_vad(audio, sr, aggressiveness=2)
print(f"Voiced frames: {voiced_mask.sum()} / {len(voiced_mask)}")
```

### Visualize

```python
from visualization import plot_pitch_vad_combined

plot_pitch_vad_combined(
    audio=result.audio,
    sr=result.sample_rate,
    pitch_times=result.timestamps,
    f0=result.f0,
    voiced_mask=result.voiced_mask,
    vad_times=result.vad_times_raw,
    vad_mask=result.vad_mask_raw,
    save_path="output.png",
)
```

---

## Tips & Tuning

**If PitchAcc50 is low but MACE is only 30–60 cents:**
- The singer is consistently slightly sharp or flat — this is a tuning issue, not a model issue
- Check `WorstMismatches` in the scorer output for the specific time points

**If too many frames are marked unvoiced:**
- Lower `--vad-mode` (try `1` or `0`) — reduces VAD aggressiveness
- Increase `--gap-fill` (try `20`) — bridges longer micro-pauses
- Try `--no-vad` to use only torchcrepe's periodicity threshold

**If too many background / breath frames are marked voiced:**
- Raise `--vad-mode` (try `3`)
- Lower `--periodicity-threshold` (try `0.15`) so torchcrepe rejects more frames

**If pitch has many octave errors:**
- Narrow the frequency range: `--fmin 100 --fmax 700` for typical singing
- The octave error rate is shown in `metrics.evaluate_all()` output

**If inference is slow:**
- Install the CUDA build of PyTorch — `--device auto` will pick it up automatically
- Reduce `--batch-size` if running out of VRAM (default is 1024)
- Use `--backend pyin` for a pure-CPU alternative (no GPU needed, slightly noisier)

**If the pitch contour looks choppy:**
- Increase `smoothing_sigma` in `FusionConfig` (try `2.0`)
- Increase `median_filter_size` (try `7`) to remove more outlier frames

---

## File Responsibilities

| File               | What to change it for                                              |
|--------------------|--------------------------------------------------------------------|
| `vad.py`           | VAD backend, smoothing logic, energy fallback threshold            |
| `pitch_wrapper.py` | Adding new pitch backends, changing torchcrepe decoder settings    |
| `alignment.py`     | Frame-rate conversion between VAD and pitch timelines              |
| `fusion.py`        | Masking strategy, gap fill, median/Gaussian smoothing parameters   |
| `inference.py`     | Pipeline config defaults, CLI flags, output format                 |
| `visualization.py` | New plot types or colour schemes                                   |
| `metrics.py`       | New evaluation metrics or threshold windows                        |
| `utils.py`         | Audio I/O backends, JSON schema, device detection                  |
| `pitch_score.py`   | Frame-level and note-level scoring formulas (existing — unchanged) |
