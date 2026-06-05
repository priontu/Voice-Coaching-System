# CLAUDE.md — Voice Coaching System

Context for Claude Code sessions on this project.

---

## Project Summary

AI-powered singing voice evaluation. Records a performance → extracts pitch/phonemes/note events → aligns against a MusicXML + TextGrid reference → computes deterministic metrics → produces a 0–100 score with category breakdown and coaching feedback.

---

## Active Modules (write code here)

### `VocalCoach_Kim/` — Main unified backend
The production 7-phase pipeline. All active development on the evaluation engine goes here.

- `configs/` — YAML config system. Load with `from configs.loader import load_model_config`
- `preprocessing/` — Audio loading and canonical 10-ms frame grid
- `inference/` — `UnifiedInferencePipeline` in `inference/pipeline.py`; CLI entry point is `inference/run_pipeline.py`
- `fusion/` — `FusedPerformanceRepresentation`: structured note/lyric events
- `reference/` — MusicXML and TextGrid parsers
- `alignment/` — Aligns fused output against reference (`AlignmentResult`)
- `metrics/` — Deterministic metric computation (`PerformanceMetricsReport`)
- `scoring/` — Weighted scoring + interpretation (`PerformanceScoreReport`, `InterpretationSummary`)
- `visualization/` — Dashboard plots (PNG, non-interactive Agg backend)
- `tests/` — 90+ pytest tests; run with `cd VocalCoach_Kim && python -m pytest tests/ -q`

### `voice_coach/` — HuggingFace-based model wrappers (Priontu)
A separate Python package providing HF-compatible wrappers for pitch, phoneme, and onset/offset models. Also contains data loading utilities for GTSinger, MusicXML, and TextGrid.

- `hf_models/` — HF extraction pipelines for each model type
- `data/` — Dataset classes, manifest builder, data collators
- `pitch/` — NanoPitch extractor

### `frontend/` — Web UI
NiceGUI web application. Entry point: `python frontend/app.py` → http://localhost:8080

- Calls `VocalCoach_Kim/inference/run_pitch.py` and `VocalCoach_Kim/scoring/pitch_score.py` as subprocesses
- **Known issue:** `SONG_REFERENCE_DIR` in `frontend/app.py:27` points to a root-level `song_references/` that doesn't exist — actual reference files are in `VocalCoach_Kim/song_references/`
- Song registry is hardcoded in `frontend/app.py`; add new songs to `SONG_REGISTRY`

---

## Experimental Modules (Priontu's development area)

`experimental/` — Iterative model versions under active development. Do not import from these in VocalCoach_Kim without explicit integration.

- `OnsetOffsetDetection_v1` … `v5` — Progressive versions of the onset/offset detector
- `PhonemeExtraction_v1` — Phoneme boundary runtime
- `PitchExtraction_v1` — NanoPitch-based pitch extraction
- `PitchExtraction_RMVPE_v1` — RMVPE-based pitch extraction with finetuning capability
- `VocalParse` — MusicXML evaluation utilities
- `test/` — Integration tests for each experimental module

---

## Legacy Code (do not import from)

`legacy/` — Original standalone model implementations, superseded by VocalCoach_Kim.

- `NoteModel_Kim/` — Original CNN+BiLSTM onset/offset detector (training scripts still usable)
- `PhonemeModel_Kim/` — Original Wav2Vec2+CTC phoneme detector
- `PitchModel_Kim/` — Original torchcrepe+WebRTC VAD pitch extractor
- `PitchModel_Dabin/` — Dabin's RMVPE pitch scoring variant

These are preserved for reference and standalone training runs, not for inference integration.

---

## Benchmarking

`benchmarks/NanoPitch_v2/` — Pitch estimation leaderboard framework. Contains team member submissions (`submissions/`) and evaluation scripts (`scripts/`). Not part of the inference pipeline.

---

## Key Configs

All pipeline behavior is controlled by YAML files in `VocalCoach_Kim/configs/`:

| File | Controls |
|------|----------|
| `pipeline.yaml` | Enable/disable modules, checkpoint paths, fusion settings |
| `system.yaml` | Device selection, logging level, default output paths |
| `pitch.yaml` | torchcrepe settings, VAD threshold |
| `phoneme.yaml` | Wav2Vec2 model ID, CTC decoder |
| `onset_offset.yaml` | CNN/BiLSTM or wav2vec2 architecture choice |
| `preprocessing.yaml` | Sample rate (16kHz), FFT size, hop length (160 = 10ms) |

Load configs with:
```python
from configs.loader import load_model_config, load_config
cfg = load_model_config("pitch")   # loads configs/pitch.yaml
```

---

## Conventions

- **No hidden weighting** — all scoring is deterministic and interpretable
- **10-ms canonical frame grid** — all models output to the same timeline
- **Modular by design** — each phase is independent; disable any module via `pipeline.yaml`
- **No model merging** — pitch, phoneme, and onset/offset stay as separate model classes
- **Type system** — shared types in `VocalCoach_Kim/utils/types.py`
- **Shared imports** — `VocalCoach_Kim/` is added to `sys.path` at the top of each runner script

---

## Common Commands

```bash
# Run all tests
cd VocalCoach_Kim && python -m pytest tests/ -q

# Validate environment
cd VocalCoach_Kim && python scripts/validate_environment.py

# Run smoke tests (no audio files needed)
cd VocalCoach_Kim && python scripts/smoke_test.py

# Full pipeline CLI
cd VocalCoach_Kim && python inference/run_pipeline.py \
    --audio path/to/audio.wav \
    --musicxml path/to/score.musicxml \
    --compute-metrics --compute-scores --plot

# Launch web UI
python frontend/app.py
```
