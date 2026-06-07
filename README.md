# Voice Coaching System

An AI-powered singing voice evaluation platform that scores vocal performances against a musical reference (MusicXML score + Praat TextGrid). The system extracts pitch, phoneme boundaries, and note onsets/offsets from a recording, aligns them against the reference, and produces deterministic, interpretable scores with coaching feedback.

---

## Directory Structure

```
Voice-Coaching-System/
├── pipeline/        # Main unified backend — 7-phase evaluation pipeline
├── hf_models/           # HuggingFace-based model wrappers and data utilities (Priontu)
├── frontend/              # NiceGUI web UI (audio upload → live scoring)
├── experimental/          # Iterative model versions under active development
├── benchmarks/            # NanoPitch_v2 pitch estimation evaluation leaderboard
├── docs/                  # Presentations and research documents
└── legacy/                # Original standalone model implementations (superseded)
```

### pipeline — 7-Phase Pipeline

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

Full architecture reference: [pipeline/setup/ARCHITECTURE.md](pipeline/setup/ARCHITECTURE.md)

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
cd pipeline
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

For detailed usage, flags, and troubleshooting see [pipeline/setup/QUICKSTART.md](pipeline/setup/QUICKSTART.md).

---

## Model Checkpoints

Place trained checkpoints in `pipeline/checkpoints/`:

```
pipeline/checkpoints/
└── best.ckpt     # wav2vec2 onset/offset model (required when enabled)
```

The phoneme model (Wav2Vec2) and pitch model (torchcrepe) download weights automatically from HuggingFace on first run.

---

## Running Tests

```bash
cd pipeline
python -m pytest tests/ -q
```

---

## Diagram

<img width="1280" height="720" alt="Slide1" src="https://github.com/user-attachments/assets/22fea440-9a28-4ece-ace7-e9616d8625d1" />

---

## Demo Video

https://github.com/user-attachments/assets/6153b164-153e-4228-891e-345a2e7c528f

---

## Contributors

| Name | Role |
|------|------|
| Kim | Unified backend (pipeline), NoteModel, PhonemeModel, PitchModel |
| Priontu | Experimental models, HF wrappers (hf_models), NanoPitch benchmarking |
| Dabin | PitchModel (RMVPE variant), pitch scoring, Frontend interface, user input flow |
