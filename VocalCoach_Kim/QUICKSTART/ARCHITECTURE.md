# VocalCoach — Architecture Reference (Phase 7)

## Overview

VocalCoach is a research-grade singing voice evaluation system. It extracts
interpretable acoustic features — phoneme boundaries, fundamental frequency
(F0), and note onsets/offsets — and scores them against a reference score or
ground-truth annotation.

Phase 1 standardized the repository structure, interfaces, configs, and
utilities. Phase 2 unified audio preprocessing and temporal synchronization.
Phase 3 adds `UnifiedInferencePipeline` — one entry point that orchestrates
all three models and returns `UnifiedInferenceResult` with all streams aligned
to the canonical 10-ms frame grid. Phase 4 adds the feature fusion layer:
structured note events, lyric/word events, phrase segmentation, voiced-region
extraction, and the canonical `FusedPerformanceRepresentation` ready for scoring.
Phase 5 adds ground-truth parsing (MusicXML + Praat TextGrid) and deterministic
prediction ↔ reference alignment. Phase 6 adds the metric computation engine:
deterministic, interpretable pitch/timing/duration/lyric metrics derived from
`AlignmentResult`, aggregated into a `PerformanceMetricsReport`. Phase 7 adds
the Scoring Engine and Musical Interpretation Layer: configurable normalization
curves, category-level scoring, confidence-weighted aggregation, and a
rule-based interpretation engine — all without LLMs or freeform generation.

---

## Repository Structure

```
VocalCoach/
│
├── api/                        # (placeholder) REST/gRPC endpoints for future deployment
├── checkpoints/                # Trained model weights (.pt files)
├── data/                       # Raw audio and annotation data
├── preprocessing/              # Centralized audio preprocessing package (Phase 2)
├── fusion/                     # Feature fusion layer (Phase 2+4)
├── reference/                  # Ground-truth parsing (Phase 5)
├── alignment/                  # Prediction ↔ reference alignment engine (Phase 5)
├── tests/                      # Unit and integration tests
│
├── configs/                    # YAML configuration files + loader
│   ├── system.yaml             # System-wide defaults (device, logging, paths)
│   ├── preprocessing.yaml      # Shared audio preprocessing parameters
│   ├── phoneme.yaml            # Phoneme module configuration
│   ├── pitch.yaml              # Pitch + VAD pipeline configuration
│   ├── onset_offset.yaml       # Note onset/offset model configuration
│   └── loader.py               # Config loading and merging utilities
│
├── utils/                      # Shared infrastructure utilities
│   ├── audio.py                # Unified audio loading, resampling, normalization
│   ├── types.py                # Shared dataclasses (PhonemeSegment, PitchFrame, NoteEvent…)
│   ├── device.py               # PyTorch device detection and assignment
│   ├── logging_utils.py        # Structured logging setup
│   ├── checkpoints.py          # Checkpoint save/load helpers
│   └── __init__.py             # Re-exports all public symbols
│
├── models/                     # Per-module model implementations
│   ├── base.py                 # BaseInferenceModel abstract interface
│   ├── phoneme/
│   │   ├── phoneme_model.py    # Wav2Vec2 + CTC pipeline + PhonemeInferenceModel
│   │   └── __init__.py
│   ├── pitch/
│   │   ├── vad.py              # WebRTC VAD wrapper (VADConfig, WebRTCVAD)
│   │   ├── pitch_wrapper.py    # torchcrepe / pYIN backends (PitchConfig, PitchModelWrapper)
│   │   ├── alignment.py        # VAD-to-pitch frame alignment utilities
│   │   ├── fusion.py           # VAD + pitch fusion and contour cleaning
│   │   ├── pipeline.py         # PitchVADPipeline + PitchInferenceModel
│   │   └── __init__.py
│   └── onset_offset/
│       ├── model.py            # OnsetOffsetModel (CNN + BiLSTM)
│       ├── spec_utils.py       # Log-mel spectrogram, peak-picking, label utilities
│       ├── detector.py         # NoteDetector + OnsetOffsetInferenceModel
│       └── __init__.py
│
├── inference/                  # Clean CLI entry points
│   ├── run_phoneme.py          # python inference/run_phoneme.py <audio.wav>
│   ├── run_pitch.py            # python inference/run_pitch.py --audio <file>
│   └── run_onset_offset.py     # python inference/run_onset_offset.py --audio <file>
│
├── metrics/                    # Evaluation metric functions (pure computation, no I/O)
│   ├── phoneme_metrics.py      # Boundary precision / recall / F1 / MAE
│   ├── pitch_metrics.py        # Frame-level pitch accuracy, MACE, RMSE; note-level accuracy
│   └── onset_offset_metrics.py # Onset / offset / duration P/R/F1/MAE
│
├── scoring/
│   └── pitch_score.py          # MusicXML loading + pitch scoring orchestration
│
├── visualization/
│   ├── pitch_viz.py            # Waveform, VAD, pitch contour, cent error plots
│   └── phoneme_viz.py          # Phoneme boundary timeline overlay
│
│   (Original model directories — preserved for backward compatibility)
├── Phoneme Model/              # Original monolithic phoneme pipeline
├── Pitch Model w VAD/          # Original pitch + VAD pipeline
├── Note Model/                 # Original onset/offset detection module
└── gtsinger/                   # GTSinger dataset subset
```

---

## Module Responsibilities

### `utils/` — Shared Infrastructure

| Module | Responsibility |
|--------|---------------|
| `audio.py` | Single entry point for all audio I/O. Loads WAV/MP3/FLAC/OGG, converts to mono, resamples to 16 kHz, normalizes. Returns numpy (`load_audio`) or torch.Tensor (`load_audio_torch`). Provides `frame_audio`, `audio_to_pcm16`, `generate_timestamps`. |
| `types.py` | Canonical dataclasses: `PhonemeSegment`, `PitchFrame`, `PitchResult`, `NoteEvent`, `AudioFeatures`, `InferenceResult`. All JSON-serializable via `asdict()`. |
| `device.py` | `get_device(preference)` — CUDA → MPS → CPU auto-selection, with explicit override support. `get_torch_device()` wraps to `torch.device`. |
| `logging_utils.py` | `setup_logging(level)` — configures root logger once at startup. `get_logger(name)` — named logger for each module. |
| `checkpoints.py` | `load_checkpoint(path, model, device)` — loads `.pt` checkpoint dict, applies `model.load_state_dict`. `save_checkpoint(path, model, ...)` — saves model + optimizer state. |

### `configs/` — Configuration System

All YAML configs are loaded via `configs/loader.py`:

```python
from configs.loader import load_model_config, load_config, merge_configs

cfg = load_model_config("pitch")   # system.yaml + pitch.yaml, deep-merged
```

- **`system.yaml`** — Device preference, logging level, path defaults.
- **`preprocessing.yaml`** — Shared audio parameters (sample_rate, n_fft, n_mels…).
- **`phoneme.yaml`** — Wav2Vec2 model name, CTC settings, post-processing thresholds.
- **`pitch.yaml`** — VAD aggressiveness, torchcrepe / pYIN parameters, fusion config.
- **`onset_offset.yaml`** — CNN/BiLSTM architecture, training hyperparameters, peak-picking thresholds.

Each model config dataclass exposes a `from_yaml(cfg)` classmethod for programmatic construction.

### `models/` — Model Implementations

#### `models/base.py` — `BaseInferenceModel`

Abstract interface that all model wrappers implement:

```python
class BaseInferenceModel(ABC):
    def load_model(self) -> None: ...   # initialize weights
    def predict(self, audio) -> Any: ... # run inference on loaded audio
    def run(self, audio_path) -> Any: ... # end-to-end from file
```

#### `models/phoneme/` — Phoneme Boundary Detection

- **Input:** WAV audio file
- **Model:** `facebook/wav2vec2-lv-60-espeak-cv-ft` (pretrained, HuggingFace)
- **Algorithm:** CTC alignment → blank-region scanning → long-segment splitting
- **Output:** `List[PhonemeSegment]` with `(phoneme, start_time, end_time, confidence)`
- **Entry class:** `PhonemeInferenceModel` / `extract_phoneme_boundaries_from_audio()`

#### `models/pitch/` — VAD + Pitch Estimation

Five-module pipeline:

```
WebRTCVAD → PitchModelWrapper → align_vad_to_pitch → fuse_vad_and_pitch → PipelineOutput
```

- **VAD:** py-webrtcvad (energy fallback when not installed)
- **Pitch backend:** torchcrepe (primary, GPU) or librosa pYIN (CPU fallback)
- **Output:** `PipelineOutput` → `pitch_data.json` consumed by `scoring/pitch_score.py`
- **Entry class:** `PitchVADPipeline` / `PitchInferenceModel`

#### `models/onset_offset/` — Note Onset/Offset Detection

- **Input:** WAV audio file → log-mel spectrogram [B, 1, n_mels, T]
- **Model:** `OnsetOffsetModel` — CNN feature extractor + BiLSTM + two MLP heads
- **Algorithm:** peak-picking on sigmoid onset/offset probability curves
- **Output:** `List[Dict]` with `{onset_time, offset_time, duration}` per note
- **Entry class:** `NoteDetector` / `OnsetOffsetInferenceModel`

### `inference/` — CLI Entry Points

All scripts share the same conventions:
- Load config from `configs/` (YAML) and override with CLI args
- Use `utils.audio.load_audio` for audio loading
- Use `utils.device.get_device` for device selection
- Use `utils.logging_utils.setup_logging` for logging

```bash
python inference/run_phoneme.py   singing.wav --output out.json --plot
python inference/run_pitch.py     --audio singing.wav --output pitch_data.json --visualize
python inference/run_onset_offset.py --checkpoint checkpoints/best.pt --audio singing.wav
```

### `metrics/` — Pure Evaluation Functions

No I/O dependencies — only numpy/math computations.

| Module | Key Functions |
|--------|--------------|
| `phoneme_metrics.py` | `compute_boundary_metrics()`, `compute_phoneme_accuracy()` |
| `pitch_metrics.py` | `compute_frame_pitch_metrics()`, `compute_note_level_pitch_correctness()` |
| `onset_offset_metrics.py` | `onset_metrics()`, `offset_metrics()`, `duration_metrics()`, `evaluate_file()` |

### `scoring/pitch_score.py`

Orchestrates pitch scoring end-to-end:
1. Load `pitch_data.json` (from `inference/run_pitch.py`)
2. Parse MusicXML reference with music21
3. Build frame-level reference F0 timeline
4. Call `metrics/pitch_metrics.py` for frame and note metrics
5. Optionally generate `visualization/pitch_viz.py` plots

### `visualization/`

| Module | Plots |
|--------|-------|
| `pitch_viz.py` | Waveform + VAD shading, pitch contour, 3-panel combined, cent error, note-level error bar chart |
| `phoneme_viz.py` | Waveform + colour-coded phoneme timeline |

---

## Shared Utilities — Migration Notes

### What Changed

| Before | After | Why |
|--------|-------|-----|
| Each model had its own `load_audio()` | `utils/audio.load_audio()` shared by all | Eliminate ~3 duplicate implementations |
| `get_best_device()` in Pitch Model `utils.py` | `utils/device.get_device()` | Single source of truth for device logic |
| No shared config system | `configs/*.yaml` + `configs/loader.py` | Consistent parameter management across models |
| Inline `logging.basicConfig()` in each script | `utils/logging_utils.setup_logging()` | Avoid duplicate root logger configuration |
| `torch.load()` called inline | `utils/checkpoints.load_checkpoint()` | Standardized error handling + logging |
| `PhonemeSegment` only in phoneme_model.py | Re-exported from `utils/types.py` | Shared type contract for future fusion |
| Visualization embedded in model files | `visualization/phoneme_viz.py`, `pitch_viz.py` | Separation of concerns |
| Metrics embedded in model files or scoring scripts | `metrics/*.py` | Pure computation, no I/O, reusable by all modules |

### What Was NOT Changed

- All model architectures (`OnsetOffsetModel`, Wav2Vec2 usage, torchcrepe)
- All prediction logic (CTC alignment, VAD, pitch fusion, peak picking)
- The `pitch_data.json` format consumed by `scoring/pitch_score.py`
- The original `Phoneme Model/`, `Pitch Model w VAD/`, `Note Model/` directories

---

## Data Flow Summary

```
Audio File (.wav)
    │
    ▼ utils/audio.py (load_audio / load_audio_torch)
    │
    ├──► models/phoneme/  → List[PhonemeSegment]
    │                             │
    │                             ▼ metrics/phoneme_metrics.py
    │                           boundary P/R/F1/MAE
    │
    ├──► models/pitch/    → PipelineOutput → pitch_data.json
    │                             │
    │                             ▼ scoring/pitch_score.py
    │                      + MusicXML → metrics/pitch_metrics.py
    │                           frame & note pitch accuracy
    │
    └──► models/onset_offset/ → List[NoteEvent]
                                      │
                                      ▼ metrics/onset_offset_metrics.py
                                    onset/offset P/R/F1/MAE
```

---

## Phase 2 — Unified Audio Preprocessing & Temporal Synchronization

### Canonical Timing Constants

```python
SAMPLE_RATE    = 16000   # Hz
HOP_LENGTH     = 160     # samples per canonical frame → 10 ms
FRAME_DURATION = 0.01    # seconds (= HOP_LENGTH / SAMPLE_RATE)
```

All frame ↔ time conversions go through `preprocessing/timestamps.py`.
Every other module imports from there — constants are never duplicated.

### Frame-Rate Mismatch (and Resolution)

| Model            | Native hop | Native fps | Resolution                              |
|------------------|:----------:|:----------:|----------------------------------------|
| torchcrepe pitch | 160        | 100 fps    | **Canonical** — no resampling needed    |
| Wav2Vec2 phoneme | 320        |  50 fps    | Segments → `segments_to_frame_labels()` |
| WebRTC VAD       | 320        |  50 fps    | `resample_mask()` nearest-neighbour     |
| CNN+BiLSTM onset | 256        | ~62.5 fps  | `resample_sequence()` linear interp     |

Phoneme boundaries are not frame-sampled; they are time-continuous segments
that are mapped to frames via `frame_overlap_ratio()` (largest-overlap wins).

### Timestamp Convention

All timestamps represent **frame centers** (not starts) unless explicitly noted:

```
frame index i → center time = i * hop_sec + hop_sec / 2
```

`canonical_timestamps(n, center=True)` is the authoritative builder.
The `center=True` convention matches torchcrepe and avoids off-by-half errors
when comparing pitch frames to phoneme segment boundaries.

### `preprocessing/` Package

```
preprocessing/
├── __init__.py          # re-exports all public symbols
├── timestamps.py        # frame↔time, align_to_grid, snap_to_frame — temporal foundation
├── framing.py           # n_frames_from_*, frame_signal, vad_frame_boundaries
├── normalization.py     # peak_normalize, rms_normalize, normalize_log_mel
├── spectrograms.py      # compute_log_mel, compute_log_mel_torch (numpy + torch)
└── audio_pipeline.py    # AudioPreprocessor — model-specific entry points
```

#### `AudioPreprocessor` — per-model entry points

```python
from preprocessing.audio_pipeline import AudioPreprocessor

pp = AudioPreprocessor(cfg)                     # pass preprocessing.yaml dict
waveform   = pp.process_for_phoneme(path)       # → 1-D torch.Tensor (CPU)
audio, sr  = pp.process_for_pitch(path)         # → np.ndarray + int
tensor, ts = pp.process_for_onset_offset(path)  # → (1,1,n_mels,T) tensor + timestamps
```

`process_for_onset_offset()` preserves the trained hop=256 by default.

### New Types in `utils/types.py`

| Type | Shape | Purpose |
|------|-------|---------|
| `MelSpectrogramFeatures` | `(n_mels, n_frames)` | Spectrogram + timestamps bundle |
| `FrameAlignedFeatures` | `(n_frames,)` per stream | Multi-model output on canonical grid |
| `TimestampedFeatureSequence` | `(N,)` or `(N, D)` | Generic named sequence |

`FrameAlignedFeatures` holds `timestamps`, `f0`, `voiced`, `onset_probs`,
`offset_probs`, and `phoneme_labels` — all at the same length so they can be
stacked or iterated frame-by-frame without further alignment.

### `fusion/alignment.py`

Extends `models/pitch/alignment.py` (boolean VAD masks only) to continuous sequences:

```python
from fusion.alignment import merge_model_outputs

aligned = merge_model_outputs(
    n_canonical=n_pitch_frames,
    pitch_times=pitch_t, f0=f0, voiced=voiced,
    onset_times=onset_t, onset_probs=onset_p,
    phoneme_segments=phoneme_segs,
)
# aligned.timestamps, aligned.f0, aligned.voiced, aligned.onset_probs,
# aligned.phoneme_labels — all length n_canonical
```

Key functions:

| Function | Use case |
|----------|----------|
| `resample_mask(src_t, mask, tgt_t)` | Boolean VAD/voiced mask → new grid |
| `resample_sequence(src_t, v, tgt_t)` | Continuous F0 / probs → new grid |
| `align_to_canonical(src_t, v, n)` | Shorthand: resample onto canonical n-frame grid |
| `segments_to_frame_labels(segs, n)` | PhonemeSegment list → per-frame string labels |
| `snap_boundary(t)` | Snap event time to nearest canonical frame center |
| `merge_model_outputs(n, ...)` | Full multi-model merge → FrameAlignedFeatures |

### Updated Data Flow (Phase 2)

```
Audio File (.wav)
    │
    ▼ preprocessing/audio_pipeline.py  (AudioPreprocessor)
    │   ├─ process_for_phoneme()    → 1-D torch.Tensor
    │   ├─ process_for_pitch()      → np.ndarray
    │   └─ process_for_onset_offset()→ 4-D tensor + timestamps
    │
    ├──► models/phoneme/   → List[PhonemeSegment]   (50fps segments)
    │
    ├──► models/pitch/     → PipelineOutput          (100fps, canonical)
    │                             timestamps, f0, voiced
    │
    └──► models/onset_offset/ → onset_probs, offset_probs  (~62.5fps)
                                      │
                                      ▼ fusion/alignment.py
                              merge_model_outputs()
                                      │
                                      ▼ FrameAlignedFeatures
                              timestamps (100fps), f0, voiced,
                              onset_probs, offset_probs, phoneme_labels
                                      │
                               scoring / visualization
```

---

## Phase 3 — Unified Multi-Model Inference Pipeline

### New Files

| File | Purpose |
|------|---------|
| `inference/pipeline.py` | `UnifiedInferencePipeline` — orchestrates all three models |
| `inference/device_manager.py` | `DeviceManager` — single device selection for all models |
| `inference/run_pipeline.py` | CLI for single-file and batch directory inference |
| `models/registry.py` | `ModelRegistry` — lazy-loading model factory registry |
| `configs/pipeline.yaml` | Unified pipeline config (enable flags, checkpoints, device) |
| `tests/test_pipeline.py` | Unit tests for pipeline orchestration (mocked models) |
| `tests/test_registry.py` | Unit tests for model registry |

### `UnifiedInferencePipeline` Execution Flow

```
Audio File (.wav)
    │
    ▼ AudioPreprocessor.process_for_pitch()   ← audio loaded ONCE here
    │                 (numpy float32 array)
    │
    ├──► PitchVADPipeline.run_from_array()   ← receives numpy array (no re-load)
    │         timestamps, f0, voiced          (100fps, hop=160)
    │
    ├──► PhonemeInferenceModel.predict()     ← receives torch.from_numpy(audio)
    │         List[PhonemeSegment]            (time-continuous, ~50fps encoder)
    │
    └──► NoteDetector.predict_probs_from_array() ← receives numpy array (no re-load)
              onset_probs, offset_probs       (~62.5fps, hop=256)
                    │
                    ▼ fusion.alignment.merge_model_outputs()
              FrameAlignedFeatures            (all streams → canonical 100fps grid)
                    │
              UnifiedInferenceResult
```

### `UnifiedInferenceResult` — Output Contract

```python
@dataclass
class UnifiedInferenceResult:
    audio_path: str
    sample_rate: int          # 16000
    hop_length: int           # 160 (canonical)
    duration_s: float

    # Raw per-model outputs (native frame rates preserved)
    phoneme_segments: Optional[List[PhonemeSegment]]
    pitch_timestamps: Optional[np.ndarray]    # (N_p,) 100fps
    f0: Optional[np.ndarray]                  # (N_p,) Hz
    voiced: Optional[np.ndarray]              # (N_p,) bool
    onset_timestamps: Optional[np.ndarray]    # (N_o,) ~62.5fps
    onset_probs: Optional[np.ndarray]         # (N_o,)
    offset_probs: Optional[np.ndarray]        # (N_o,)
    note_events: Optional[List[NoteEvent]]

    # Temporally aligned on canonical 100fps grid
    aligned: Optional[FrameAlignedFeatures]
    metadata: Dict                            # elapsed_s, device, enabled
```

### `ModelRegistry` — Lazy Loading

```python
registry = ModelRegistry()          # default factories pre-registered
registry.load("pitch")              # instantiates + calls load_model()
registry.load("phoneme")            # cached after first call
model = registry.get("pitch")       # returns cached instance
registry.unload("phoneme")          # drop from cache (free memory)
```

Custom models can be registered without touching existing code:
```python
registry.register("my_model", factory_fn)
registry.load("my_model", checkpoint_path="...")
```

### `DeviceManager` — Single Device Decision

```python
dm = DeviceManager(preference="auto")  # CUDA → MPS → CPU
dm.device_str    # "cuda:0" | "mps" | "cpu"
dm.torch_device  # torch.device(...)
dm.move(tensor)  # ensure tensor is on the managed device
```

One `DeviceManager` per pipeline; its `device_str` is passed to all models so
no model ever makes its own device choice.

### Usage

Single file:
```python
pipeline = UnifiedInferencePipeline.from_config_file("configs/pipeline.yaml")
result   = pipeline.predict("singing.wav")

print(result.duration_s)            # 3.52
print(len(result.phoneme_segments)) # 47
print(result.aligned.n_frames)      # 352  (100fps × 3.52s)
```

Batch directory:
```bash
python inference/run_pipeline.py --input_dir dataset/ --export-json
```

Selective modules:
```bash
python inference/run_pipeline.py --audio song.wav --no-phoneme
```

### Phase 3 Migration Notes

| Before (Phase 2) | After (Phase 3) |
|------------------|----------------|
| Call each model's `run(path)` separately | `pipeline.predict(path)` orchestrates all |
| Audio loaded 1–3× (once per model) | Audio loaded exactly once |
| No unified result type | `UnifiedInferenceResult` with aligned view |
| Device set per-model via YAML | Single `DeviceManager` propagated to all |
| No batch support | `predict_batch()` / `predict_directory()` |

`NoteDetector.predict_probs_from_array(audio_np, sr)` was added as a new method
(no existing code modified) so the pipeline can pass the pre-loaded array
directly without re-reading the audio file.

---

## Phase 4 — Feature Fusion & Musical Event Construction

### New Files

| File | Purpose |
|------|---------|
| `fusion/note_events.py` | `build_note_events()` — onset/offset probs → rich NoteEvent list |
| `fusion/lyric_events.py` | `build_lyric_events()` — PhonemeSegments → LyricEvent / WordEvent |
| `fusion/event_alignment.py` | Cross-event overlap analysis, voiced regions, phrase segmentation |
| `fusion/validation.py` | `ValidationReport` + structural consistency checks |
| `visualization/fusion_viz.py` | Matplotlib panels for all fused event streams |
| `tests/test_note_events.py` | Unit tests for note event construction (45+ tests) |
| `tests/test_lyric_events.py` | Unit tests for lyric/word construction (35+ tests) |
| `tests/test_event_alignment.py` | Unit tests for alignment, overlap, regions (50+ tests) |

### New Types in `utils/types.py`

| Type | Purpose |
|------|---------|
| `LyricEvent` | One phoneme in lyrical context; links to word and note by index |
| `WordEvent` | Proximity-grouped phonemes; contains `List[LyricEvent]` |
| `PhraseEvent` | Consecutive notes below inter-note gap threshold |
| `TemporalRegion` | Labeled time interval (voiced / unvoiced / silence) |
| `FusedPerformanceRepresentation` | Canonical downstream object for scoring (Phase 5+) |

`NoteEvent` is expanded with optional Phase 4 fields (fully backward-compatible):
`pitch_hz`, `pitch_midi`, `pitch_stability`, `voiced_fraction`,
`confidence`, `onset_confidence`, `offset_confidence`,
`phoneme_labels`, `lyric_text`, `note_idx`.

### Event Construction Flow

```
Pitch + Onset probs + Offset probs + Voiced mask
              ↓  fusion/note_events.build_note_events()
         List[NoteEvent]
         (onset_time, offset_time, pitch_hz, pitch_midi,
          pitch_stability, voiced_fraction, confidence)

PhonemeSegments + Timestamps
              ↓  fusion/lyric_events.build_lyric_events()
         List[LyricEvent]  +  List[WordEvent]
         (1:1 from segments)   (proximity-grouped)

              ↓  fusion/event_alignment.annotate_notes_with_phonemes()
         NoteEvent.phoneme_labels, NoteEvent.lyric_text  ← back-annotated

              ↓  fusion/event_alignment.annotate_words_with_notes()
         WordEvent.note_idx, LyricEvent.note_idx  ← back-annotated

              ↓  fusion/event_alignment.build_phrase_events()
         List[PhraseEvent]  (notes grouped by silence gap)

              ↓  fusion/event_alignment.build_voiced_regions()
         List[TemporalRegion]  (voiced / unvoiced contiguous spans)

              ↓  FusedPerformanceRepresentation
         (canonical downstream object for scoring)
```

### `FusedPerformanceRepresentation` — Output Contract

```python
@dataclass
class FusedPerformanceRepresentation:
    audio_path: str
    duration_s: float
    sample_rate: int         # 16000
    hop_length: int          # 160 (canonical)

    # Canonical-grid arrays (all length n_frames, 100fps)
    timestamps: np.ndarray   # (n_frames,) seconds
    f0: np.ndarray           # (n_frames,) Hz; 0 = unvoiced
    voiced: np.ndarray       # (n_frames,) bool

    # Structured events (sorted by start_time)
    note_events: List[NoteEvent]
    lyric_events: List[LyricEvent]
    word_events: List[WordEvent]
    phrase_events: List[PhraseEvent]
    voiced_regions: List[TemporalRegion]

    # Raw phoneme segments (preserved for downstream)
    phoneme_segments: Optional[List[PhonemeSegment]]

    # Metadata
    alignment_metadata: Dict   # e.g. note_phoneme_coverage stats
    inference_metadata: Dict   # copied from UnifiedInferenceResult.metadata
```

### Pipeline Integration

Enable fusion via `configs/pipeline.yaml`:

```yaml
fusion:
  enabled: true
  min_note_duration_ms: 50
  onset_threshold: 0.5
  offset_threshold: 0.5
  word_gap_ms: 100
  phrase_gap_s: 0.5
  min_region_duration_ms: 20
  overlap_threshold: 0.5
  boundary_snap_ms: 20
```

Then `result.fused` is populated after `pipeline.predict()`:

```python
pipeline = UnifiedInferencePipeline.from_config_file("configs/pipeline.yaml")
result   = pipeline.predict("singing.wav")

fused = result.fused               # FusedPerformanceRepresentation
print(fused.n_notes)               # 12
print(len(fused.word_events))      # 8
print(fused.note_events[0].pitch_hz)   # 330.4 Hz
print(fused.note_events[0].lyric_text) # "AH-EH"
```

### Updated Data Flow (Phase 4)

```
Audio File (.wav)
    │
    ▼ AudioPreprocessor (loaded ONCE)
    │
    ├──► PitchVADPipeline     → timestamps, f0, voiced (100fps)
    │
    ├──► PhonemeInferenceModel → List[PhonemeSegment]
    │
    └──► NoteDetector         → onset_probs, offset_probs (~62.5fps)
                    │
                    ▼ fusion/alignment.merge_model_outputs()
             FrameAlignedFeatures  (all streams → 100fps canonical grid)
                    │
                    ▼ [fusion.enabled=true]
             ┌─────────────────────────────────┐
             │ build_note_events()             │
             │ build_lyric_events()            │
             │ annotate_notes_with_phonemes()  │
             │ annotate_words_with_notes()     │
             │ build_phrase_events()           │
             │ build_voiced_regions()          │
             │ validate_fused_representation() │
             └─────────────────────────────────┘
                    │
             FusedPerformanceRepresentation
             (canonical input for scoring / feedback)
```

### Validation

`fusion/validation.py` provides `validate_fused_representation(fused)` which
returns a `ValidationReport` classifying all issues as `'warning'` (non-blocking)
or `'error'` (blocks scoring). Checks include:

- Negative or zero note/lyric/word/phrase durations
- Onset ≥ offset time
- Events extending beyond audio duration
- Overlapping notes (warning) or regions
- Canonical array length mismatch (error)
- Word span ≠ constituent phoneme span (warning)
- Phrase note indices out of range (error)

### Visualization

```python
from visualization.fusion_viz import plot_fused_timeline

fig = plot_fused_timeline(fused, figsize=(14, 10))
fig.savefig("debug.png", dpi=150, bbox_inches="tight")
```

4-panel layout: pitch contour (F0 + voiced shading), note bars (MIDI-coloured),
phoneme segments, word segments — all sharing a common time axis.

### Phase 4 Migration Notes

| Before (Phase 3) | After (Phase 4) |
|------------------|----------------|
| `NoteEvent` has only onset/offset/duration | Expanded with pitch stats and phoneme linkage (backward-compatible) |
| `fusion/` has only `alignment.py` | 4 new modules: `note_events`, `lyric_events`, `event_alignment`, `validation` |
| Pipeline produces `UnifiedInferenceResult` only | Also produces `FusedPerformanceRepresentation` when `fusion.enabled=true` |
| No structured word/phrase events | `WordEvent`, `PhraseEvent`, `TemporalRegion` types added |
| Visualization covers pitch and phonemes | `fusion_viz.py` adds fused timeline, note–phoneme alignment views |

---

## Phase 5 — Ground Truth Parsing & Reference Alignment

### New Files

| File | Purpose |
|------|---------|
| `reference/__init__.py` | Package re-exports |
| `reference/musicxml_parser.py` | `parse_musicxml()` — MusicXML → ReferenceNote list via music21 |
| `reference/textgrid_parser.py` | `parse_textgrid()` — Praat TextGrid → phoneme/word lists |
| `reference/reference_builder.py` | `build_reference_representation()` — combines both sources |
| `reference/validation.py` | Structural validation of ReferencePerformanceRepresentation |
| `alignment/__init__.py` | Package re-exports |
| `alignment/alignment_utils.py` | Pure temporal/pitch utility functions |
| `alignment/reference_alignment.py` | `align_performance()` — greedy overlap-based matching |
| `visualization/reference_viz.py` | Predicted vs. reference comparison plots |
| `tests/test_musicxml_parser.py` | MusicXML parser tests (synthetic XML in temp files) |
| `tests/test_textgrid_parser.py` | TextGrid parser tests (synthetic TextGrid in temp files) |
| `tests/test_reference_alignment.py` | Alignment engine tests (pure Python objects) |

### New Types in `utils/types.py`

| Type | Purpose |
|------|---------|
| `ReferenceNote` | One note from MusicXML: onset/offset in seconds, MIDI pitch, lyric, measure/beat |
| `ReferencePhoneme` | One phoneme from TextGrid: timed label with word back-link |
| `ReferenceWord` | One word from TextGrid: timed label with phoneme index list |
| `ReferencePhrase` | Note grouping by silence gap (mirrors PhraseEvent) |
| `ReferencePerformanceRepresentation` | Canonical reference object: notes + phonemes + words + phrases |
| `NoteAlignmentMatch` | One predicted↔reference note pair: overlap, onset/offset deviation, pitch deviation |
| `PhonemeAlignmentMatch` | One predicted↔reference phoneme pair: overlap, label match flag |
| `WordAlignmentMatch` | One predicted↔reference word pair: overlap, onset deviation |
| `AlignmentResult` | Full alignment output: all match lists + unmatched indices + metadata |

### MusicXML Parsing Flow

```
score.xml  (MusicXML 3.x)
    │
    ▼  music21.converter.parse()
    │
    ├── MetronomeMark  → tempo_bpm  (fallback: 120 BPM)
    ├── TimeSignature  → time_signature (numerator, denominator)
    ├── Key            → key_signature string
    └── notesAndRests  (flat, all parts merged)
            │
            ▼  beats → seconds conversion (60 / tempo_bpm)
            │
            ▼  tied note merging (greedy left-to-right)
            │
     List[ReferenceNote]
     (onset_time, offset_time, pitch_midi, pitch_hz, pitch_name,
      lyric, measure, beat, is_rest, is_tied, note_idx)
```

### TextGrid Parsing Flow

```
annotation.TextGrid  (Praat interval tier format)
    │
    ▼  backend resolution (praatio → textgrid → built-in plain-text)
    │
    ├── phoneme tier  →  intervals filtered (skip silence)
    │                      │
    │                      ▼  ReferencePhoneme objects
    │                         (phoneme, start_time, end_time, phoneme_idx)
    │
    └── word tier  →  intervals filtered (skip silence)
                        │
                        ▼  ReferenceWord objects
                           (text, start_time, end_time, phoneme_indices, word_idx)
                           ↑ back-annotated: phonemes within span get word_idx
```

### Reference Builder

`build_reference_representation()` combines both sources:

```python
from reference.reference_builder import build_reference_representation

ref = build_reference_representation(
    musicxml_path="score.xml",
    textgrid_path="annotation.TextGrid",
    phoneme_tier="phonemes",
    word_tier="words",
    phrase_gap_s=0.5,
)
# ref.notes      — from MusicXML
# ref.phonemes   — from TextGrid
# ref.words      — from TextGrid
# ref.phrases    — segmented by silence gap (mirrors prediction-side logic)
# ref.tempo_bpm, ref.time_signature, ref.key_signature
```

Either source is optional — the function raises only when both are absent.

### Alignment Pipeline

```
FusedPerformanceRepresentation      ReferencePerformanceRepresentation
  (predicted events)                  (ground-truth events)
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
           alignment/reference_alignment.align_performance()
                       │
           Greedy overlap matrix (descending overlap order)
           One-to-one matching with configurable thresholds
                       │
                       ▼
                AlignmentResult
                  note_matches       List[NoteAlignmentMatch]
                  phoneme_matches    List[PhonemeAlignmentMatch]
                  word_matches       List[WordAlignmentMatch]
                  unmatched_pred_*   unmatched predicted indices
                  unmatched_ref_*    unmatched reference indices
                  alignment_metadata  precision, recall, mean deviations
```

### Alignment Utilities (`alignment/alignment_utils.py`)

| Function | Returns | Description |
|----------|---------|-------------|
| `overlap_duration(a, b)` | `float` | Temporal intersection in seconds |
| `overlap_fraction_of_a(a, b)` | `float` | Fraction of A covered by overlap with B |
| `iou(a, b)` | `float` | Intersection-over-union |
| `onset_deviation(pred, ref)` | `float` | Signed onset error (positive = late) |
| `offset_deviation(pred, ref)` | `float` | Signed offset error |
| `nearest_match(query, candidates)` | `(int, float)` | Index + distance |
| `pitch_deviation_cents(pred_hz, ref_hz)` | `Optional[float]` | Cents deviation |
| `pitch_deviation_semitones(pred_hz, ref_hz)` | `Optional[float]` | Semitone deviation |

### Pipeline Integration (Phase 5)

Pass reference paths to `pipeline.predict()`:

```python
pipeline = UnifiedInferencePipeline.from_config_file("configs/pipeline.yaml")

# With reference alignment (fusion must also be enabled)
result = pipeline.predict(
    "singing.wav",
    musicxml_path="score.xml",
    textgrid_path="annotation.TextGrid",
)

ref       = result.reference   # ReferencePerformanceRepresentation
alignment = result.alignment   # AlignmentResult (None if fusion disabled)

print(alignment.n_note_matches)       # 11
print(alignment.note_precision)       # 0.85
print(alignment.note_recall)          # 0.92
print(alignment.note_matches[0].onset_deviation_s)   # -0.032 (32ms early)
print(alignment.note_matches[0].pitch_deviation_cents) # 15.4 (slightly sharp)
```

Reference parsing runs even when fusion is disabled; alignment is skipped with
a log warning if `result.fused is None`.

### Validation (`reference/validation.py`)

Validates `ReferencePerformanceRepresentation` before alignment:

```python
from reference.validation import validate_reference_representation

report = validate_reference_representation(reference)
# report.is_valid       → False if any 'error' issues found
# report.n_errors       → count
# report.n_warnings     → count
```

Checks include:
- Negative timestamps (error)
- `end_time ≤ start_time` (error)
- Overlapping reference phonemes (error)
- Overlapping non-rest notes (warning)
- `pitch_midi` outside [0, 127] (warning)
- Tempo consistency: last note offset vs. declared duration (warning)
- Missing notes / phonemes tier (warning)

### Updated Data Flow (Phase 5)

```
Audio File (.wav)
    │
    ▼ UnifiedInferencePipeline.predict(
           audio_path,
           musicxml_path=...,   ← optional
           textgrid_path=...    ← optional
      )
    │
    ├── PitchVADPipeline → f0, voiced (100fps)
    ├── PhonemeInferenceModel → List[PhonemeSegment]
    └── NoteDetector → onset_probs, offset_probs
                │
                ▼ fusion/alignment.merge_model_outputs()
          FrameAlignedFeatures  (100fps canonical)
                │
                ▼ [fusion.enabled=true]
          FusedPerformanceRepresentation
                │
                ▼ [musicxml_path or textgrid_path provided]
                │
          ┌─────────────────────────────────────────────┐
          │  reference/musicxml_parser.parse_musicxml() │
          │  reference/textgrid_parser.parse_textgrid() │
          │  reference/reference_builder.build_ref()    │
          │  reference/validation.validate_ref()        │
          └──────────────────────┬──────────────────────┘
                                 │
                 ReferencePerformanceRepresentation
                                 │
                ┌────────────────▼────────────────────┐
                │ alignment/reference_alignment        │
                │   align_notes()                      │
                │   align_phonemes()                   │
                │   align_words()                      │
                └────────────────┬────────────────────┘
                                 │
                          AlignmentResult
                                 │
                    UnifiedInferenceResult
                    .reference = ReferencePerformanceRepresentation
                    .alignment = AlignmentResult
```

### Visualization (`visualization/reference_viz.py`)

```python
from visualization.reference_viz import plot_alignment_summary

fig = plot_alignment_summary(
    fused, reference, alignment,
    suptitle="singing.wav vs score.xml",
    save_path="debug_alignment.png",
)
```

4-panel layout:
1. **Note timeline overlay** — predicted (blue/red) vs. reference (orange/darkorange) bars; grey lines connect matched pairs
2. **Onset deviation bar chart** — signed error per matched note; red = late, blue = early
3. **Duration scatter** — predicted vs. reference duration; identity line for reference
4. **F0 contour vs. reference pitch** — predicted F0 (blue) + reference MIDI pitch as orange step segments

### Phase 5 Migration Notes

| Before (Phase 4) | After (Phase 5) |
|------------------|----------------|
| `pipeline.predict(audio_path)` | `pipeline.predict(audio_path, musicxml_path=..., textgrid_path=...)` |
| No reference parsing | `reference/` package: MusicXML + TextGrid parsers |
| `UnifiedInferenceResult` has no reference field | Adds `.reference` and `.alignment` (both Optional, None when not provided) |
| No prediction↔reference alignment | `alignment/` package: greedy overlap-based one-to-one matching |
| Phase 4 scoring used `scoring/pitch_score.py` (legacy) | Phase 5 prepares `AlignmentResult` for structured metric computation |
| `fusion/validation.py` validates prediction side | `reference/validation.py` validates reference side (separate concern) |

---

## Future Integration Path (Phase 6+)

Phase 5 delivers `AlignmentResult` — the canonical input for metric computation:

1. **Metric computation** (`metrics/`): `AlignmentResult.note_matches` contains
   `onset_deviation_s`, `offset_deviation_s`, `pitch_deviation_cents` per note —
   pass directly to precision/recall/F1/MAE calculations with no further alignment.

2. **Scoring** (`scoring/`): Aggregate `AlignmentResult` statistics into a
   per-phrase or per-performance score. Use `AlignmentResult.note_precision` and
   `note_recall` as starting signal; weight by pitch accuracy from
   `NoteAlignmentMatch.pitch_deviation_cents`.

3. **Feedback generation** (`feedback/`): `AlignmentResult.unmatched_pred_notes`
   identifies missed notes (predicted but not in score); `unmatched_ref_notes`
   identifies skipped notes (in score but not sung). Use `PhraseEvent` boundaries
   from `FusedPerformanceRepresentation` for per-phrase feedback messages.

4. **Shared encoder** (`models/shared_encoder.py`): `BaseInferenceModel` and
   the registry pattern allow dropping in a shared Wav2Vec2/HuBERT trunk
   without touching the pipeline or downstream code.

5. **API** (`api/`): `UnifiedInferencePipeline.predict()` already accepts
   `musicxml_path`, `textgrid_path`, and `compute_metrics` — expose them as
   optional form fields in the HTTP endpoint for score-aware evaluation.

---

## Phase 6 — Metric Computation Engine

### Objectives

Phase 6 computes deterministic, interpretable singing evaluation metrics from
the aligned prediction ↔ reference structures produced in Phase 5. No weighted
final scores are computed here; the output (`PerformanceMetricsReport`) is
designed to be consumed by a future scoring/feedback layer (Phase 7+).

### New files

| File | Purpose |
|------|---------|
| `metrics/pitch_metrics.py` | Phase 6 pitch functions (added to existing file) |
| `metrics/timing_metrics.py` | Onset, offset, IOI timing metrics |
| `metrics/duration_metrics.py` | Note duration error and ratio metrics |
| `metrics/lyric_metrics.py` | Phoneme boundary and word alignment metrics |
| `metrics/reporting.py` | Aggregation engine → `PerformanceMetricsReport` |
| `metrics/validation.py` | Sanity checks → `MetricValidationReport` |
| `visualization/metrics_viz.py` | Six metric visualization plots |
| `tests/test_pitch_metrics.py` | ~30 pitch metric tests |
| `tests/test_timing_metrics.py` | ~25 timing metric tests |
| `tests/test_duration_metrics.py` | ~20 duration metric tests |
| `tests/test_lyric_metrics.py` | ~25 lyric metric tests |
| `tests/test_reporting.py` | ~20 reporting + validation tests |

### New types (utils/types.py)

| Type | Description |
|------|-------------|
| `MetricBreakdown` | Per-event detail: `event_idx`, `value`, `label`, `metadata` |
| `PitchMetrics` | Pitch accuracy, RMSE, MACE, per-note breakdown |
| `TimingMetrics` | Onset/offset error, timing accuracy, IOI MAE |
| `DurationMetrics` | Duration error, ratio, relative error, per-note breakdown |
| `LyricMetrics` | Phoneme boundary error, word alignment, label match rate |
| `PerformanceMetricsReport` | Top-level aggregation of all four categories |

`UnifiedInferenceResult` gains one new Optional field:
```python
metrics: Optional[PerformanceMetricsReport] = None   # Phase 6
```

### Metric computation data flow

```
AlignmentResult
        │
        ├── NoteAlignmentMatch list ──► pitch_metrics.py ──► PitchMetrics
        │                          ──► timing_metrics.py ──► TimingMetrics
        │    + FusedPerformance    ──► duration_metrics.py ─► DurationMetrics
        │    + ReferencePerformance
        │
        └── PhonemeAlignmentMatch  ──► lyric_metrics.py ──► LyricMetrics
            WordAlignmentMatch
                    │
                    ▼
            reporting.build_metrics_report()
                    │
                    ▼
          PerformanceMetricsReport
                    │
            validation.validate_metrics_report()
                    │
                    ▼
          MetricValidationReport
```

### Metric definitions

#### Pitch metrics (`metrics/pitch_metrics.py`)

| Metric | Formula | Unit |
|--------|---------|------|
| `pitch_accuracy` | `mean(|dev| ≤ τ)` over matched notes with pitch data | fraction |
| `pitch_rmse_cents` | `√(mean(dev²))` | cents |
| `mace_cents` | `mean(|dev|)` | cents (MACE) |
| `mean_pitch_deviation_cents` | `mean(dev)` — positive = sharp | cents |
| `note_pitch_accuracy` | alias for `pitch_accuracy` | fraction |

`dev` = `NoteAlignmentMatch.pitch_deviation_cents`; positive = predicted is sharper.

#### Timing metrics (`metrics/timing_metrics.py`)

| Metric | Formula | Unit |
|--------|---------|------|
| `mean_onset_error_ms` | `mean(onset_dev)` — positive = late | ms |
| `std_onset_error_ms` | `std(onset_dev)` | ms |
| `mean_abs_onset_error_ms` | `mean(|onset_dev|)` | ms |
| `median_onset_error_ms` | median of onset_dev | ms |
| `mean_offset_error_ms` | `mean(offset_dev)` | ms |
| `timing_accuracy` | `mean(|onset_dev| ≤ τ)` | fraction |
| `ioi_mae_ms` | `mean(|pred_IOI − ref_IOI|)` by position | ms |

#### Duration metrics (`metrics/duration_metrics.py`)

| Metric | Formula | Unit |
|--------|---------|------|
| `mean_duration_error_s` | `mean(pred_dur − ref_dur)` — positive = too long | s |
| `mean_abs_duration_error_s` | `mean(|pred_dur − ref_dur|)` | s |
| `mean_duration_ratio` | `mean(pred_dur / ref_dur)` — 1.0 = perfect | dimensionless |
| `mean_relative_duration_error` | `mean(|error| / ref_dur)` — tempo-independent | fraction |

#### Lyric metrics (`metrics/lyric_metrics.py`)

| Metric | Formula | Unit |
|--------|---------|------|
| `mean_phoneme_boundary_error_ms` | `mean(onset_dev)` over phoneme matches | ms |
| `mean_abs_phoneme_boundary_error_ms` | `mean(|onset_dev|)` | ms |
| `phoneme_overlap_accuracy` | fraction of phoneme matches with `overlap_fraction ≥ 0.5` | fraction |
| `word_alignment_accuracy` | `|matched_words| / |total_ref_words|` | fraction |
| `label_match_rate` | fraction of phoneme matches where `label_match=True` | fraction |

### Pipeline integration

```python
result = pipeline.predict(
    "singing.wav",
    musicxml_path="score.xml",     # Phase 5: parses reference
    textgrid_path="labels.TextGrid",
    compute_metrics=True,           # Phase 6: computes PerformanceMetricsReport
)

# Metrics are on result.metrics
print(result.metrics.pitch.pitch_accuracy)
print(result.metrics.timing.mean_abs_onset_error_ms)
print(result.metrics.lyric.label_match_rate)
```

`compute_metrics=True` can also be set permanently via `metrics.enabled: true`
in `configs/pipeline.yaml`. Metrics are only computed when `result.alignment`
is not None (i.e. both reference parsing and alignment succeeded).

### Validation

`metrics/validation.py` checks the following invariants:
- No NaN or Inf in any numeric field
- `pitch_accuracy`, `timing_accuracy`, `note_precision`, `note_recall`, 
  `word_alignment_accuracy`, `label_match_rate` are in [0, 1]
- Absolute errors and RMSE are non-negative
- Duration ratio is non-negative
- Warnings issued for `n_evaluated=0` fields with non-None values

### Visualization (`visualization/metrics_viz.py`)

| Function | Output |
|----------|--------|
| `plot_pitch_deviation` | Per-note pitch deviation bar chart |
| `plot_onset_deviation` | Per-note onset timing error bar chart (blue=early, red=late) |
| `plot_duration_comparison` | Predicted vs. reference duration scatter with identity line |
| `plot_phoneme_timing` | Per-phoneme boundary error bar chart |
| `plot_metric_heatmap` | Normalised pitch/timing/duration heatmap |
| `plot_metrics_summary` | Five-panel figure combining all plots |

All plots use the non-interactive `Agg` backend; pass `save_path` to write PNG/PDF.

### Phase 6 migration notes

| Changed | Details |
|---------|---------|
| `utils/types.py` | +`MetricBreakdown`, `PitchMetrics`, `TimingMetrics`, `DurationMetrics`, `LyricMetrics`, `PerformanceMetricsReport`; `UnifiedInferenceResult` gains `metrics` field |
| `metrics/pitch_metrics.py` | Phase 1 functions unchanged; Phase 6 functions appended |
| `metrics/__init__.py` | Now re-exports `build_metrics_report`, `validate_metrics_report` |
| `inference/pipeline.py` | `predict()` gains `compute_metrics: bool = False`; `_run_metrics()` method added; step 8 in execution order |
| `configs/pipeline.yaml` | `metrics:` section added with `enabled`, `pitch.cents_tolerance`, `timing.onset_tolerance_ms`, `lyric.phoneme_tolerance_ms` |

### Future Phase 7+ integration

`PerformanceMetricsReport` is the input contract for the scoring and feedback
generation layer. Fields designed for downstream use:

- `pitch.pitch_accuracy` / `pitch.mace_cents` → pitch score component
- `timing.mean_abs_onset_error_ms` → timing score component
- `duration.mean_relative_duration_error` → duration score (tempo-independent)
- `lyric.word_alignment_accuracy` + `lyric.label_match_rate` → lyric score
- `note_precision` / `note_recall` → note detection score
- `*.per_note` / `*.per_phoneme` breakdowns → per-event feedback

---

## Phase 7 — Scoring Engine & Musical Interpretation Layer

### Objectives

Phase 7 transforms the raw `PerformanceMetricsReport` from Phase 6 into
deterministic, interpretable performance scores and category-level summaries.
No LLMs, no freeform text generation, no hidden weighting.

```
PerformanceMetricsReport
         ↓
  Score Normalization          scoring/normalization.py
         ↓
  Category Scoring             scoring/pitch_scoring.py
                               scoring/timing_scoring.py
                               scoring/duration_scoring.py
                               scoring/lyric_scoring.py
         ↓
  Overall Performance Score   scoring/performance_scoring.py
         ↓
  Interpretation Summary      scoring/interpretation.py
```

### New Files

| File | Responsibility |
|------|---------------|
| `scoring/normalization.py` | `bounded_score`, `gaussian_penalty`, `piecewise_score`, `normalize_metric` — deterministic normalization curves |
| `scoring/pitch_scoring.py` | `compute_intonation_score`, `compute_pitch_stability_score`, `compute_pitch_score` → `CategoryScore` |
| `scoring/timing_scoring.py` | `compute_rhythm_stability_score`, `compute_timing_score` → `CategoryScore` |
| `scoring/duration_scoring.py` | `compute_phrase_duration_score`, `compute_duration_score` → `CategoryScore` |
| `scoring/lyric_scoring.py` | `compute_phoneme_timing_score`, `compute_lyric_clarity_score` → `CategoryScore` |
| `scoring/performance_scoring.py` | `build_performance_score_report` — confidence-weighted aggregation → `PerformanceScoreReport` |
| `scoring/interpretation.py` | `build_interpretation_summary` — rule-based levels and messages → `InterpretationSummary` |
| `scoring/validation.py` | `validate_score_report` — numerical sanity checks → `ScoreValidationReport` |
| `visualization/scoring_viz.py` | `plot_category_radar`, `plot_score_breakdown`, `plot_timing_penalty`, `plot_pitch_scoring_overlay`, `plot_performance_dashboard` |
| `tests/test_normalization.py` | Unit tests for all normalization functions |
| `tests/test_pitch_scoring.py` | Unit tests for pitch scoring |
| `tests/test_timing_scoring.py` | Unit tests for timing scoring |
| `tests/test_performance_scoring.py` | Unit tests for overall scoring and validation |
| `tests/test_interpretation.py` | Unit tests for interpretation engine |

### New Types (`utils/types.py`)

| Type | Description |
|------|-------------|
| `ScoreBreakdown` | Per-component scored detail: `component`, `raw_value`, `score ∈ [0,100]`, `weight`, `confidence` |
| `CategoryScore` | Aggregate score for one category (pitch/timing/duration/lyric): `score`, `confidence`, `components`, `n_evaluated` |
| `PerformanceScoreReport` | Full scoring report: four `CategoryScore` fields + `overall_score` + `weights_used` |
| `InterpretationSummary` | Rule-based interpretation: `overall_level`, `strengths`, `weaknesses`, `category_levels` |

`UnifiedInferenceResult` gains two new optional fields:
- `scores: Optional[PerformanceScoreReport]` — populated when `compute_scores=True`
- `interpretation: Optional[InterpretationSummary]` — populated together with scores

### Normalization Strategy

All raw metric values are normalized to `[0, 100]` via one of four curves:

| Mode | Formula | Use case |
|------|---------|---------|
| `bounded` | linear clamp `[lower=best, upper=worst] → [100, 0]` | direct fractions |
| `gaussian` | `100 · exp(−x²/2σ²)` | smooth decay from 0 |
| `piecewise` | piecewise-linear through `(x, score)` breakpoints | domain-tuned curves |
| `threshold` | 100 if `x ≤ threshold` else 0 | binary pass/fail |

Category scoring uses **piecewise** curves tuned to musical perceptual boundaries:
- MACE 0¢ → 100, 50¢ → 75, 100¢ → 50, 200¢ → 0 (one semitone = 100¢)
- Onset MAE 0ms → 100, 50ms → 75, 100ms → 50, 200ms → 0
- Relative duration error 0 → 100, 0.2 → 75, 0.5 → 50, 1.0 → 0

### Category Scoring

Each category scoring function takes its Phase 6 metrics object and returns a `CategoryScore`:

| Category | Input | Components | Default weights |
|----------|-------|-----------|----------------|
| Pitch | `PitchMetrics` | accuracy, intonation (MACE), stability (RMSE) | 0.50 / 0.30 / 0.20 |
| Timing | `TimingMetrics` | accuracy, onset MAE, rhythm stability (IOI+std) | 0.50 / 0.30 / 0.20 |
| Duration | `DurationMetrics` | relative error, ratio deviation, phrase consistency | 0.60 / 0.20 / 0.20 |
| Lyric | `LyricMetrics` | word accuracy, overlap, label match, boundary timing | 0.35 / 0.25 / 0.25 / 0.15 |

### Confidence-Weighted Aggregation

Every `ScoreBreakdown` carries a `confidence ∈ [0, 1]` derived from `n_evaluated`:
`confidence = min(1.0, n_evaluated / 3.0)`. Categories with zero evaluated events
contribute `confidence=0` and are excluded from the aggregate.

The overall score is:

```
overall = Σ(category_score · nominal_weight · confidence)
        / Σ(nominal_weight · confidence)
```

`weights_used` in `PerformanceScoreReport` reports the effective (post-normalization)
weights actually applied, enabling full explainability.

### Interpretation Engine

`build_interpretation_summary()` assigns one of four levels to each evaluated category
and to the overall performance using configurable thresholds:

| Level | Default threshold |
|-------|-----------------|
| `excellent` | score ≥ 90 |
| `good` | score ≥ 75 |
| `fair` | score ≥ 55 |
| `needs_work` | score < 55 |

Strength messages are emitted for `excellent` and `good` categories; weakness messages
for `fair` and `needs_work`. All messages are looked up from a static rule table —
there is no freeform generation or LLM usage.

### Pipeline Integration

```python
# Phase 7 scoring + interpretation in one call
result = pipeline.predict(
    "singing.wav",
    musicxml_path="score.xml",
    compute_metrics=True,
    compute_scores=True,   # ← new Phase 7 parameter
)

print(result.scores.overall_score)          # e.g. 78.4
print(result.interpretation.overall_level)  # e.g. "good"
print(result.interpretation.strengths)      # ["Good pitch accuracy overall"]
print(result.interpretation.weaknesses)     # ["Minor timing inconsistencies detected"]
```

Or via `configs/pipeline.yaml`:
```yaml
scoring:
  enabled: true
  weights:
    pitch: 0.40
    timing: 0.30
    duration: 0.15
    lyric: 0.15
```

### Validation

`scoring/validation.py` checks every `PerformanceScoreReport` for:
- Non-finite scores (NaN / Inf) → **error**
- Scores outside `[0, 100]` → **error**
- Confidence outside `[0, 1]` → **error**
- Component weights ≤ 0 → **error**
- `n_evaluated=0` with non-zero score → **warning**
- `overall_score=None` → **warning**
- `weights_used` sum ≠ 1.0 → **warning**

### Phase 7 Migration Notes

| Symbol | Change |
|--------|--------|
| `utils/types.py` | +`ScoreBreakdown`, `CategoryScore`, `PerformanceScoreReport`, `InterpretationSummary`; `UnifiedInferenceResult` gains `scores` and `interpretation` fields |
| `scoring/__init__.py` | Now re-exports `build_performance_score_report`, `build_interpretation_summary`, `validate_score_report` |
| `inference/pipeline.py` | `predict()` gains `compute_scores: bool = False`; `_run_scoring()` method added; step 9 in execution order (metadata is now step 10) |
| `configs/pipeline.yaml` | `scoring:` section added with `enabled`, `weights`, per-category sub-configs, and `interpretation` thresholds |

### Phase 8+ Integration Contract

`PerformanceScoreReport` and `InterpretationSummary` are the input contracts for
the feedback generation layer:

- `interpretation.overall_level` → coaching session tone (Phase 8)
- `interpretation.strengths` + `interpretation.weaknesses` → natural-language coaching prompts (Phase 8)
- `scores.pitch_score.components` → per-aspect pitch coaching detail (Phase 8)
- `scores.overall_score` → progress tracking and gamification (Phase 9+)
