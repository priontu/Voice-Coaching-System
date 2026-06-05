# Voice Coaching System

An AI-powered singing voice evaluation platform that scores vocal performances against a musical reference (MusicXML score + Praat TextGrid). The system extracts pitch, phoneme boundaries, and note onsets/offsets from a recording, aligns them against the reference, and produces deterministic, interpretable scores with coaching feedback.

---

## Directory Structure

```
Voice-Coaching-System/
├── VocalCoach_Kim/        # Main unified backend — 7-phase evaluation pipeline
├── voice_coach/           # HuggingFace-based model wrappers and data utilities (Priontu)
├── frontend/              # NiceGUI web UI (audio upload → live scoring)
├── experimental/          # Iterative model versions under active development
├── benchmarks/            # NanoPitch_v2 pitch estimation evaluation leaderboard
├── docs/                  # Presentations and research documents
└── legacy/                # Original standalone model implementations (superseded)
```

### VocalCoach_Kim — 7-Phase Pipeline

The production backend. Each phase builds on the last:

| Phase | Directory | Responsibility |
|-------|-----------|----------------|
| 1 | `utils/`, `configs/` | Shared utilities, YAML config system |
| 2 | `preprocessing/` | Audio loading, resampling, 10-ms frame grid |
| 3 | `inference/` | Unified multi-model inference (`UnifiedInferencePipeline`) |
| 4 | `fusion/` | Structured note/lyric event representation |
| 5 | `reference/`, `alignment/` | MusicXML + TextGrid parsing and alignment |
| 6 | `metrics/` | Deterministic metric computation |
| 7 | `scoring/`, `visualization/` | Weighted scoring, interpretation, dashboards |

Full architecture reference: [VocalCoach_Kim/setup/ARCHITECTURE.md](VocalCoach_Kim/setup/ARCHITECTURE.md)

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU (CUDA 12.1):
```bash
pip install torch==2.3.0+cu121 torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 2. Validate your environment

```bash
cd VocalCoach_Kim
python scripts/validate_environment.py
```

### 3. Run the pipeline (CLI)

Pitch + phoneme only:
```bash
python inference/run_pipeline.py --audio path/to/recording.wav
```

Full pipeline with reference scoring:
```bash
python inference/run_pipeline.py \
    --audio      path/to/recording.wav \
    --musicxml   path/to/reference.musicxml \
    --textgrid   path/to/reference.TextGrid \
    --compute-metrics --compute-scores --plot \
    --output_dir outputs/
```

### 4. Launch the web UI

```bash
python frontend/app.py
# Open http://localhost:8080
```

For detailed usage, flags, and troubleshooting see [VocalCoach_Kim/setup/QUICKSTART.md](VocalCoach_Kim/setup/QUICKSTART.md).

---

## Model Checkpoints

Place trained checkpoints in `VocalCoach_Kim/checkpoints/`:

```
VocalCoach_Kim/checkpoints/
└── best.ckpt     # wav2vec2 onset/offset model (required when enabled)
```

The phoneme model (Wav2Vec2) and pitch model (torchcrepe) download weights automatically from HuggingFace on first run.

---

## Running Tests

```bash
cd VocalCoach_Kim
python -m pytest tests/ -q
```

---

## Contributors

| Name | Role |
|------|------|
| Kim | Unified backend (VocalCoach_Kim), NoteModel, PhonemeModel, PitchModel |
| Priontu | Experimental models, HF wrappers (voice_coach), NanoPitch benchmarking |
| Dabin | PitchModel (RMVPE variant), pitch scoring |
