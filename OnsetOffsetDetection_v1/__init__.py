"""Conformer-style onset/offset detection for singing voice coaching.

Imports are intentionally lazy so data utilities do not pay Lightning's import
cost unless the training module is actually requested.
"""

__all__ = [
    "OnsetOffsetConformer",
    "OnsetOffsetFeatureConfig",
    "OnsetOffsetLabelConfig",
    "OnsetOffsetLightningModule",
    "OnsetOffsetModelConfig",
    "OnsetOffsetTrainingConfig",
]


def __getattr__(name: str):
    if name in {"OnsetOffsetConformer", "OnsetOffsetModelConfig"}:
        from .model import OnsetOffsetConformer, OnsetOffsetModelConfig

        return {"OnsetOffsetConformer": OnsetOffsetConformer, "OnsetOffsetModelConfig": OnsetOffsetModelConfig}[name]
    if name in {"OnsetOffsetFeatureConfig", "OnsetOffsetLabelConfig"}:
        from .preprocessing import OnsetOffsetFeatureConfig, OnsetOffsetLabelConfig

        return {
            "OnsetOffsetFeatureConfig": OnsetOffsetFeatureConfig,
            "OnsetOffsetLabelConfig": OnsetOffsetLabelConfig,
        }[name]
    if name in {"OnsetOffsetLightningModule", "OnsetOffsetTrainingConfig"}:
        from .lightning_module import OnsetOffsetLightningModule, OnsetOffsetTrainingConfig

        return {
            "OnsetOffsetLightningModule": OnsetOffsetLightningModule,
            "OnsetOffsetTrainingConfig": OnsetOffsetTrainingConfig,
        }[name]
    raise AttributeError(name)
