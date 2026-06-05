# Legacy Model Implementations

These directories contain the original standalone model implementations from early development. They are **superseded by `pipeline/`** and are preserved for reference and standalone training runs.

| Directory | Replaced by |
|-----------|-------------|
| `NoteModel_Kim/` | `pipeline/models/onset_offset/` |
| `PhonemeModel_Kim/` | `pipeline/models/phoneme/` |
| `PitchModel_Kim/` | `pipeline/models/pitch/` |
| `PitchModel_Dabin/` | `pipeline/models/pitch/` (RMVPE variant) |

Do not import from these directories in new code. Use `pipeline/` instead.

Training scripts (`train.py`, `evaluate.py`) in each directory remain functional for retraining individual models outside the unified pipeline.
