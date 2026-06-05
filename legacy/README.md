# Legacy Model Implementations

These directories contain the original standalone model implementations from early development. They are **superseded by `VocalCoach_Kim/`** and are preserved for reference and standalone training runs.

| Directory | Replaced by |
|-----------|-------------|
| `NoteModel_Kim/` | `VocalCoach_Kim/models/onset_offset/` |
| `PhonemeModel_Kim/` | `VocalCoach_Kim/models/phoneme/` |
| `PitchModel_Kim/` | `VocalCoach_Kim/models/pitch/` |
| `PitchModel_Dabin/` | `VocalCoach_Kim/models/pitch/` (RMVPE variant) |

Do not import from these directories in new code. Use `VocalCoach_Kim/` instead.

Training scripts (`train.py`, `evaluate.py`) in each directory remain functional for retraining individual models outside the unified pipeline.
