# VocalCoach — Quick Start Guide

End-to-end singing voice evaluation from audio file to scored performance report.  
This guide gets you from a clean Python install to a working pipeline in under 15 minutes.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup](#2-environment-setup)
3. [Repository Setup](#3-repository-setup)
4. [Validate Your Environment](#4-validate-your-environment)
5. [Run Smoke Tests](#5-run-smoke-tests)
6. [Quick Start — Single File](#6-quick-start--single-file)
7. [Full Pipeline — With Reference Alignment](#7-full-pipeline--with-reference-alignment)
8. [Batch Evaluation](#8-batch-evaluation)
9. [Visualization](#9-visualization)
10. [Example Outputs](#10-example-outputs)
11. [Optional Flags Reference](#11-optional-flags-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU (optional) | — | CUDA 11.8+ / 4 GB VRAM |
| Disk | 4 GB | 8 GB (for model weights) |
| OS | Windows 10 / Linux | Windows 11 / Ubuntu 22.04 |
| ffmpeg | required for MP3/FLAC | any recent version |

> **Note:** All inference runs on CPU if no GPU is available.
> Pitch estimation on CPU takes approximately 1–3× real-time for typical recordings.

---

## 2. Environment Setup

### 2.1 Create a virtual environment

```powershell
# Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2.2 Install dependencies (CPU)

```powershell
pip install -r requirements.txt
```

### 2.3 Install dependencies (GPU — CUDA 12.1)

```powershell
pip install -r requirements.txt
pip install torch==2.3.0+cu121 torchaudio==2.3.0+cu121 `
    --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8 (older drivers):

```powershell
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 `
    --index-url https://download.pytorch.org/whl/cu118
```

Check your CUDA version:

```powershell
nvcc --version
```

### 2.4 Install ffmpeg (Windows)

ffmpeg is required to load MP3 and FLAC audio files.

```powershell
# Using winget (Windows 11)
winget install --id=Gyan.FFmpeg -e

# Using Chocolatey
choco install ffmpeg

# Verify installation
ffmpeg -version
```

> If you use only WAV files, ffmpeg is optional.

---

## 3. Repository Setup

### 3.1 Expected directory structure

```
VocalCoach/
├── configs/                 # Pipeline configuration YAML files
│   ├── pipeline.yaml        # Main pipeline config
│   ├── system.yaml          # System-wide defaults
│   ├── pitch.yaml           # Pitch model config
│   └── phoneme.yaml         # Phoneme model config
├── inference/
│   ├── pipeline.py          # Core UnifiedInferencePipeline class
│   └── run_pipeline.py      # CLI entry point
├── scoring/                 # Phase 7 scoring engine
├── metrics/                 # Phase 6 metric computation
├── visualization/           # Plot generation
├── scripts/
│   ├── validate_environment.py
│   ├── smoke_test.py
│   └── batch_evaluate.py
├── checkpoints/             # Model weight files (see §3.2)
├── samples/                 # Your audio files go here
├── references/              # MusicXML / TextGrid reference files
├── outputs/                 # Generated JSON results and plots
├── requirements.txt
├── requirements-gpu.txt
├── run_demo.ps1             # One-click demo (PowerShell)
└── run_demo.bat             # One-click demo (CMD)
```

### 3.2 Checkpoint directory

Model checkpoints are downloaded automatically on first run for the phoneme model
(from HuggingFace). The pitch model uses `torchcrepe` which downloads weights
automatically.

The onset/offset detector requires a manually trained checkpoint (disabled by
default in `configs/pipeline.yaml`):

```
checkpoints/
└── onset_offset_best.pt    # (optional — only if onset/offset is enabled)
```

To enable onset/offset detection:

```yaml
# configs/pipeline.yaml
pipeline:
  enable_onset_offset: true
  checkpoints:
    onset_offset: "checkpoints/onset_offset_best.pt"
```

### 3.3 Sample files directory

Place your WAV files in `samples/`:

```
samples/
├── example.wav         # 16kHz mono WAV recommended
├── aria_practice.wav
└── ...
```

Place reference files in `references/` with the same stem as the audio:

```
references/
├── example.musicxml    # MusicXML score
├── example.TextGrid    # Praat TextGrid (phoneme boundaries)
└── ...
```

---

## 4. Validate Your Environment

Before running inference, verify all dependencies are installed:

```powershell
py scripts\validate_environment.py
```

Expected output on a healthy system:

```
========================================================
  VocalCoach Environment Validation
========================================================

--- Python ---
  [PASS] Python 3.11.8

--- Required packages ---
  [PASS] torch 2.3.0
  [PASS] torchaudio 2.3.0
  [PASS] numpy 1.26.4
  [PASS] librosa 0.10.1
  [PASS] pyyaml 6.0.1
  [PASS] matplotlib 3.8.3

--- Optional packages ---
  [PASS] music21 9.3.0
  [PASS] praatio 6.0.0
  [PASS] torchcrepe 0.0.22
  [PASS] transformers 4.40.1
  [PASS] scipy 1.12.0
  [PASS] soundfile 0.12.1
  [PASS] pytest 7.4.4

--- GPU ---
  [PASS] CUDA available — NVIDIA GeForce RTX 3080 (10 GB)

--- System tools ---
  [PASS] ffmpeg — ffmpeg version 6.1.1

--- Config files ---
  [PASS] configs/pipeline.yaml
  [PASS] configs/system.yaml
  [PASS] configs/pitch.yaml

--- Directories ---
  [WARN] samples/  not found — will be created on first use
  [WARN] references/  not found — will be created on first use

========================================================
  All checks passed. Environment is ready.
========================================================
```

If any `[FAIL]` lines appear, install the missing packages before proceeding.

---

## 5. Run Smoke Tests

Verify integration correctness with synthetic data (no audio files needed):

```powershell
py scripts\smoke_test.py
```

For verbose output with tracebacks on failure:

```powershell
py scripts\smoke_test.py --verbose
```

Expected output:

```
============================================================
  VocalCoach Smoke Test Suite
============================================================

  Tier 1 — Core imports
    PASS  import: utils.types Phase 7 types  (12ms)
    PASS  import: configs.loader  (3ms)
    PASS  import: preprocessing.audio_pipeline  (45ms)
    PASS  import: scoring.normalization  (2ms)
    PASS  import: scoring.pitch_scoring  (2ms)
    PASS  import: scoring.timing_scoring  (2ms)
    PASS  import: scoring.duration_scoring  (2ms)
    PASS  import: scoring.lyric_scoring  (2ms)
    PASS  import: scoring.performance_scoring  (3ms)
    PASS  import: scoring.interpretation  (2ms)
    PASS  import: scoring.validation  (2ms)
    PASS  import: metrics.reporting  (8ms)
    PASS  import: visualization.scoring_viz  (120ms)
    PASS  import: inference.pipeline (no model load)  (15ms)

  Tier 2 — Config loading
    PASS  config: pipeline.yaml loads and has required sections  (5ms)
    PASS  config: system.yaml loads  (2ms)
    PASS  config: merge_configs deep-merges correctly  (0ms)

  Tier 3 — Normalization
    PASS  normalization: bounded_score lower/upper extremes  (0ms)
    PASS  normalization: gaussian_penalty zero deviation → 100  (0ms)
    PASS  normalization: piecewise_score interpolates correctly  (0ms)
    PASS  normalization: non-finite input → 0  (0ms)
    PASS  normalization: invalid mode raises ValueError  (0ms)

  Tier 4 — Scoring (synthetic)
    PASS  scoring: compute_pitch_score returns valid CategoryScore  (1ms)
    PASS  scoring: compute_timing_score returns valid CategoryScore  (1ms)
    PASS  scoring: build_performance_score_report full metrics  (2ms)
    PASS  scoring: build_performance_score_report pitch-only  (1ms)
    PASS  scoring: validate_score_report no errors on valid report  (1ms)
    PASS  scoring: weights_used sum to 1.0  (1ms)
    PASS  scoring: to_dict is JSON-serializable  (1ms)

  Tier 5 — Interpretation
    PASS  interpretation: overall_level assigned  (1ms)
    PASS  interpretation: excellent for score=92  (0ms)
    PASS  interpretation: needs_work for score=40  (0ms)
    PASS  interpretation: deterministic on same input  (1ms)
    PASS  interpretation: to_dict JSON-serializable  (0ms)

  Tier 6 — Pipeline init
    PASS  pipeline: initializes from dict overrides  (8ms)
    PASS  pipeline: from_config_file reads pipeline.yaml  (5ms)

============================================================
  All 36 smoke tests passed.
============================================================
```

> If any test fails, run `py scripts\smoke_test.py --verbose` for the full traceback.

---

## 6. Quick Start — Single File

### 6.1 Minimal inference (pitch + phoneme only)

```powershell
py .\inference\run_pipeline.py --audio ".\samples\example.wav"
```

Console output:

```
────────────────────────────────────────────────────────
  File:      example.wav
  Duration:  12.34s
  Device:    cuda:0
  Elapsed:   2.41s
────────────────────────────────────────────────────────
  Pitch:     1234 frames, 78.3% voiced
             F0 175–440 Hz (med 261 Hz)
  Phonemes:  87 segments
────────────────────────────────────────────────────────
```

### 6.2 Export result to JSON

```powershell
py .\inference\run_pipeline.py `
    --audio ".\samples\example.wav" `
    --export-json `
    --output_dir ".\outputs\"
```

Output file: `outputs\example_unified.json`

### 6.3 One-click demo script

```powershell
.\run_demo.ps1 -AudioFile ".\samples\example.wav"
```

Or CMD:

```cmd
run_demo.bat "samples\example.wav"
```

---

## 7. Full Pipeline — With Reference Alignment

The full pipeline runs: Inference → Fusion → Reference Alignment → Metrics → Scoring → Interpretation.

### 7.1 Requirements

Enable fusion in `configs/pipeline.yaml` (required for reference alignment):

```yaml
fusion:
  enabled: true
```

### 7.2 Full pipeline command

```powershell
py .\inference\run_pipeline.py `
    --audio      ".\samples\example.wav" `
    --musicxml   ".\references\example.musicxml" `
    --textgrid   ".\references\example.TextGrid" `
    --compute-metrics `
    --compute-scores `
    --export-json `
    --plot `
    --output_dir ".\outputs\"
```

Console output:

```
────────────────────────────────────────────────────────
  File:      example.wav
  Duration:  12.34s
  Device:    cuda:0
  Elapsed:   4.87s
────────────────────────────────────────────────────────
  Pitch:     1234 frames, 78.3% voiced
             F0 175–440 Hz (med 261 Hz)
  Phonemes:  87 segments
  Aligned:   1234 canonical frames (100fps)
  Ref align: 42 note matches
  Pitch acc: 81.0%  MACE 22.4¢
  Timing:    76.2%  MAE 31.5ms
  Score:     79.3/100  pitch=83  timing=76  dur=81  lyr=72
  Level:     GOOD
  Strengths: Good pitch accuracy overall
  Improve:   Timing consistency could be more precise
────────────────────────────────────────────────────────
  JSON:      outputs\example_unified.json
  Plot:      outputs\example_dashboard.png
```

### 7.3 MusicXML only (no TextGrid)

```powershell
py .\inference\run_pipeline.py `
    --audio    ".\samples\example.wav" `
    --musicxml ".\references\example.musicxml" `
    --compute-metrics `
    --compute-scores
```

### 7.4 TextGrid only (no MusicXML)

```powershell
py .\inference\run_pipeline.py `
    --audio    ".\samples\example.wav" `
    --textgrid ".\references\example.TextGrid" `
    --compute-metrics `
    --compute-scores
```

---

## 8. Batch Evaluation

### 8.1 Basic batch (inference only)

```powershell
py .\inference\run_pipeline.py `
    --input_dir ".\samples\" `
    --export-json `
    --output_dir ".\outputs\"
```

### 8.2 Batch with scoring

```powershell
py .\scripts\batch_evaluate.py `
    --input-dir   ".\dataset\audio\" `
    --reference-dir ".\dataset\references\" `
    --output-dir  ".\outputs\batch\" `
    --compute-metrics `
    --compute-scores `
    --export-json
```

File matching convention:

```
dataset/audio/        →   dataset/references/
  01_aria.wav               01_aria.musicxml
  01_aria.wav               01_aria.TextGrid
  02_folk.wav               02_folk.musicxml
```

Console output:

```
Batch evaluation: 3 file(s) in dataset\audio

[1/3] 01_aria.wav
  MusicXML:  01_aria.musicxml
  Elapsed:   5.12s
  Score:     82.4/100  [pit=86  tim=79  dur=84  lyr=78]
  Level:     GOOD

[2/3] 02_folk.wav
  MusicXML:  02_folk.musicxml
  Elapsed:   4.89s
  Score:     71.3/100  [pit=73  tim=68  dur=75  lyr=69]
  Level:     FAIR

[3/3] 03_pop.wav
  Elapsed:   3.44s
  Score:     N/A  (no reference files found)

────────────────────────────────────────────────────────
  Batch complete — 3 file(s) in 13.5s
  Average score:  76.9/100
  Score range:    71.3 – 82.4
────────────────────────────────────────────────────────
  Summary JSON:   outputs\batch\batch_summary.json
```

---

## 9. Visualization

Plots are generated when `--plot` is passed and scoring results are available.

### 9.1 Dashboard (4-panel)

```powershell
py .\inference\run_pipeline.py `
    --audio    ".\samples\example.wav" `
    --musicxml ".\references\example.musicxml" `
    --compute-metrics --compute-scores --plot
```

Output: `outputs\example_dashboard.png`

The dashboard contains:
- **Top-left:** Radar chart — overall category scores
- **Top-right:** Score breakdown bars per category
- **Bottom-left:** Pitch score component breakdown
- **Bottom-right:** Timing score component breakdown

### 9.2 Generate plots from existing JSON

```python
import json
from utils.types import PerformanceScoreReport
from visualization.scoring_viz import plot_performance_dashboard

with open("outputs/example_unified.json") as f:
    data = json.load(f)

# Reconstruct report (if scores were exported)
# plot_performance_dashboard(score_report, save_path="outputs/example_dashboard.png")
```

---

## 10. Example Outputs

### 10.1 JSON output structure

`outputs/example_unified.json` (truncated for readability):

```json
{
  "audio_path": "samples/example.wav",
  "sample_rate": 16000,
  "hop_length": 160,
  "duration_s": 12.34,
  "f0": [0.0, 0.0, 261.3, 263.1, 264.0, "..."],
  "voiced": [false, false, true, true, true, "..."],
  "phoneme_segments": [
    {"phoneme": "AH", "start_time": 0.21, "end_time": 0.34, "confidence": 0.91},
    {"phoneme": "V",  "start_time": 0.34, "end_time": 0.41, "confidence": 0.87},
    "..."
  ],
  "metrics": {
    "audio_path": "samples/example.wav",
    "pitch": {
      "n_evaluated": 42,
      "pitch_accuracy": 0.810,
      "mace_cents": 22.4,
      "pitch_rmse_cents": 28.1,
      "per_note": ["..."]
    },
    "timing": {
      "n_evaluated": 42,
      "timing_accuracy": 0.762,
      "mean_abs_onset_error_ms": 31.5,
      "median_onset_error_ms": 24.0
    },
    "duration": {
      "n_evaluated": 42,
      "mean_relative_duration_error": 0.12,
      "mean_duration_ratio": 0.94
    },
    "lyric": {
      "n_evaluated": 87,
      "word_alignment_accuracy": 0.84,
      "phoneme_overlap_accuracy": 0.79,
      "label_match_rate": 0.81
    }
  },
  "scores": {
    "audio_path": "samples/example.wav",
    "overall_score": 79.3,
    "weights_used": {
      "pitch": 0.4,
      "timing": 0.3,
      "duration": 0.15,
      "lyric": 0.15
    },
    "pitch_score": {
      "category": "pitch",
      "score": 83.2,
      "confidence": 1.0,
      "n_evaluated": 42,
      "components": [
        {"component": "pitch_accuracy", "raw_value": 0.810, "score": 81.0, "weight": 0.5},
        {"component": "intonation",     "raw_value": 22.4,  "score": 85.5, "weight": 0.3},
        {"component": "stability",      "raw_value": 28.1,  "score": 84.1, "weight": 0.2}
      ]
    },
    "timing_score": {
      "category": "timing",
      "score": 76.1,
      "confidence": 1.0,
      "n_evaluated": 42,
      "components": [
        {"component": "timing_accuracy", "raw_value": 0.762, "score": 76.2, "weight": 0.5},
        {"component": "onset_mae",       "raw_value": 31.5,  "score": 76.6, "weight": 0.3},
        {"component": "rhythm_stability","raw_value": 28.0,  "score": 75.2, "weight": 0.2}
      ]
    },
    "duration_score": {
      "category": "duration",
      "score": 81.4,
      "confidence": 1.0,
      "n_evaluated": 42
    },
    "lyric_score": {
      "category": "lyric",
      "score": 72.1,
      "confidence": 1.0,
      "n_evaluated": 87
    }
  },
  "interpretation": {
    "audio_path": "samples/example.wav",
    "overall_level": "good",
    "category_levels": {
      "pitch": "good",
      "timing": "fair",
      "duration": "good",
      "lyric": "fair"
    },
    "strengths": [
      "Good pitch accuracy overall",
      "Well-controlled note durations"
    ],
    "weaknesses": [
      "Timing consistency could be more precise",
      "Lyric synchronization needs attention"
    ]
  },
  "metadata": {
    "elapsed_s": 4.87,
    "device": "cuda:0",
    "enabled": {
      "pitch": true,
      "phoneme": true,
      "onset_offset": false,
      "fusion": true,
      "reference_alignment": true,
      "metrics": true,
      "scores": true
    }
  }
}
```

### 10.2 Batch summary JSON

`outputs/batch/batch_summary.json`:

```json
{
  "total_elapsed_s": 13.45,
  "files": [
    {
      "file": "01_aria.wav",
      "elapsed_s": 5.12,
      "overall_score": 82.4,
      "level": "good"
    },
    {
      "file": "02_folk.wav",
      "elapsed_s": 4.89,
      "overall_score": 71.3,
      "level": "fair"
    },
    {
      "file": "03_pop.wav",
      "elapsed_s": 3.44,
      "overall_score": null,
      "level": null
    }
  ]
}
```

---

## 11. Optional Flags Reference

### `inference/run_pipeline.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` / `-a` | — | Single WAV/MP3/FLAC file (required in single-file mode) |
| `--input_dir` / `-d` | — | Directory of audio files (batch mode) |
| `--musicxml` | None | MusicXML reference score (enables alignment) |
| `--textgrid` | None | Praat TextGrid file (enables phoneme alignment) |
| `--compute-metrics` | off | Compute PerformanceMetricsReport |
| `--compute-scores` | off | Compute PerformanceScoreReport + interpretation |
| `--export-json` | off | Write JSON results to `--output_dir` |
| `--plot` | off | Generate scoring dashboard plots |
| `--output_dir` / `-o` | `outputs/` | Output directory |
| `--device` | auto | `auto` \| `cpu` \| `cuda` \| `cuda:N` |
| `--no-pitch` | off | Disable pitch+VAD inference |
| `--no-phoneme` | off | Disable phoneme inference |
| `--no-onset-offset` | off | Disable onset/offset inference |
| `--config` | None | Path to YAML config file override |
| `--verbose` / `-v` | off | Enable DEBUG logging |

### `scripts/batch_evaluate.py`

| Flag | Description |
|------|-------------|
| `--input-dir` | Directory of audio files |
| `--reference-dir` | Directory of matching reference files |
| `--output-dir` | Output directory (default: `outputs/`) |
| `--pattern` | Glob pattern (default: `*.wav`) |
| `--compute-metrics` | Compute metrics after alignment |
| `--compute-scores` | Compute scores + interpretation |
| `--export-json` | Export per-file JSON + batch summary |

---

## 12. Troubleshooting

### 12.1 CUDA not detected

```
[WARN] CUDA not available — will use CPU
```

**Fix:**

```powershell
# Check torch CUDA build
py -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# If False: reinstall with correct CUDA version
pip install torch==2.3.0+cu121 torchaudio==2.3.0+cu121 `
    --index-url https://download.pytorch.org/whl/cu121
```

Verify NVIDIA driver: `nvidia-smi`

---

### 12.2 ModuleNotFoundError on startup

```
ModuleNotFoundError: No module named 'librosa'
```

**Fix:**

```powershell
pip install -r requirements.txt
```

If running from VSCode or an IDE, ensure the correct virtual environment is selected.

---

### 12.3 Audio file not found

```
Error: [Errno 2] No such file or directory: 'samples/example.wav'
```

**Fix:** Create the `samples/` directory and add a WAV file, or pass an absolute path:

```powershell
py .\inference\run_pipeline.py --audio "C:\Users\you\recordings\song.wav"
```

---

### 12.4 ffmpeg not found (MP3 / FLAC loading fails)

```
NoBackendError: No backend found for file 'song.mp3'
```

**Fix:**

```powershell
winget install --id=Gyan.FFmpeg -e
# Then restart PowerShell and verify:
ffmpeg -version
```

For Linux:

```bash
sudo apt-get install ffmpeg
```

---

### 12.5 Onset/offset checkpoint missing

```
[ERROR] Onset/offset failed: checkpoint not found
```

**Fix:** The onset/offset model is disabled by default. Either:

1. Keep it disabled (default):

```yaml
# configs/pipeline.yaml
pipeline:
  enable_onset_offset: false
```

2. Or provide a trained checkpoint:

```yaml
pipeline:
  enable_onset_offset: true
  checkpoints:
    onset_offset: "checkpoints/onset_offset_best.pt"
```

---

### 12.6 Scoring returns `None` overall score

The overall score is `None` when no category has data. This happens when:

- `--compute-metrics` was not passed
- Reference alignment failed (fusion not enabled)
- The audio contained no voiced frames

**Debug workflow:**

```powershell
# Step 1: Check that metrics computed
py .\inference\run_pipeline.py --audio song.wav --musicxml song.xml --compute-metrics --verbose

# Step 2: Verify fusion is enabled
# configs/pipeline.yaml → fusion.enabled: true

# Step 3: Check alignment result in JSON
# Look for "alignment" key in the JSON output
```

---

### 12.7 Reference alignment skipped

```
[INFO] Reference parsed but alignment skipped (fusion not enabled)
```

**Fix:** Enable fusion in `configs/pipeline.yaml`:

```yaml
fusion:
  enabled: true
```

---

### 12.8 Sample rate mismatch

```
RuntimeError: Expected sample rate 16000, got 44100
```

**Fix:** The pipeline resamples automatically. If this error appears, it indicates a preprocessing issue:

```powershell
# Convert audio to 16kHz mono WAV with ffmpeg
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

---

### 12.9 Matplotlib / plotting failures on headless server

```
cannot connect to X server
```

**Fix:** The visualization module uses the Agg (non-interactive) backend, but if the issue persists:

```python
import matplotlib
matplotlib.use("Agg")   # must be called before any other matplotlib import
```

Or set the environment variable:

```powershell
$env:MPLBACKEND = "Agg"
```

---

### 12.10 NaN metric values

```
[WARN] nan values detected in PitchMetrics
```

**Cause:** Usually means no voiced note pairs were matched during alignment.

**Debug:**

```powershell
# Check voiced percentage in output
py .\inference\run_pipeline.py --audio song.wav --verbose
# Look for: "78.3% voiced" — if 0%, pitch extraction failed

# Check note match count
# Look for: "Ref align: 0 note matches"
```

---

### 12.11 Windows long-path issues

```
OSError: [Errno 36] File name too long
```

**Fix:** Enable long paths in Windows:

```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

### 12.12 Slow inference on CPU

Expected CPU runtimes (approximate):

| Stage | 10s audio | 30s audio |
|-------|-----------|-----------|
| Pitch (CPU) | 12–20s | 35–60s |
| Phoneme (CPU) | 5–8s | 15–25s |
| Scoring | <1s | <1s |

**Speed up options:**

```powershell
# GPU inference (if available)
py .\inference\run_pipeline.py --audio song.wav --device cuda

# CPU only, disable unused modules
py .\inference\run_pipeline.py --audio song.wav --no-onset-offset
```

---

## Debugging Workflow

When the pipeline produces unexpected results, follow this checklist:

```
1. py scripts\validate_environment.py        # check environment
2. py scripts\smoke_test.py --verbose        # check imports + logic
3. py inference\run_pipeline.py --verbose    # trace pipeline steps
4. inspect JSON output:                      # check which fields are None
   - metrics=None → alignment failed or --compute-metrics not passed
   - scores=None  → metrics were not populated
   - overall_score=None → no category data available
5. check pipeline.yaml settings:
   - fusion.enabled: true    (required for alignment)
   - metrics.enabled / scoring.enabled (or pass --compute-metrics / --compute-scores)
```

---

*VocalCoach — Modular Singing Voice Evaluation System, Phase 7*
