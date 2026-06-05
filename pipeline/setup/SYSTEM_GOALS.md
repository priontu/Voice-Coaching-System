Goal:
Build an interpretable singing voice evaluation system.

Current completed modules:
- phoneme boundary detection
- pitch + VAD
- onset/offset detection

Current architecture:
- separate models
- separate preprocessing pipelines

Phase 1 objective:
- unify repository structure
- standardize utilities
- standardize configs
- prepare for future integration

IMPORTANT:
Do NOT merge models yet.
Do NOT rewrite working model logic unnecessarily.
Prioritize modularity and shared infrastructure.

Phase 2 Goal:
Unify preprocessing and timestamp synchronization across all models.

Do NOT redesign model architectures.
Do NOT merge models.
Preserve existing inference behavior.
Prioritize timestamp consistency and reusable preprocessing.

Phase 3 Goal:
Create a unified orchestration pipeline that loads and runs all models through a single synchronized inference API.

Do NOT merge models.
Do NOT redesign architectures.
Do NOT implement scoring yet.
Preserve modularity and compatibility.

Phase 4 Goal:
Transform synchronized low-level inference outputs into structured musical and lyrical event representations.

Do NOT implement scoring yet.
Do NOT redesign models.
Preserve modular modularity and deterministic synchronization.

Phase 5 Goal:
Parse and align reference musical/lyrical ground truth data against fused inference outputs.

Do NOT implement scoring yet.
Do NOT redesign model architectures.
Preserve deterministic synchronization and modularity.

Phase 6 Goal:
Compute deterministic, interpretable singing evaluation metrics from aligned prediction and reference structures.

Do NOT implement final weighted scoring yet.
Do NOT implement coaching feedback yet.
Preserve modularity and deterministic evaluation behavior.

Phase 7 Goal:
Transform low-level evaluation metrics into deterministic, interpretable singing performance scores and category evaluations.

Do NOT implement natural-language coaching yet.
Do NOT redesign models.
Preserve modularity, determinism, and interpretability.