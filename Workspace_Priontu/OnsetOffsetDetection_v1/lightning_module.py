from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

try:
    import lightning as L
except ImportError as exc:
    raise RuntimeError("lightning is required for OnsetOffsetDetection_v1") from exc

from .model import OnsetOffsetConformer, OnsetOffsetModelConfig
from .preprocessing import OnsetOffsetBatchProcessor, OnsetOffsetFeatureConfig, OnsetOffsetLabelConfig


@dataclass
class OnsetOffsetTrainingConfig:
    model: OnsetOffsetModelConfig = field(default_factory=OnsetOffsetModelConfig)
    features: OnsetOffsetFeatureConfig = field(default_factory=OnsetOffsetFeatureConfig)
    labels: OnsetOffsetLabelConfig = field(default_factory=OnsetOffsetLabelConfig)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    onset_pos_weight: float = 8.0
    offset_pos_weight: float = 8.0
    active_pos_weight: float = 1.5
    max_audio_sec: float | None = 30.0


class OnsetOffsetLightningModule(L.LightningModule):
    def __init__(self, config: OnsetOffsetTrainingConfig = OnsetOffsetTrainingConfig()):
        super().__init__()
        self.config_obj = config
        self.save_hyperparameters(asdict(config))
        self.model = OnsetOffsetConformer(config.model)
        self.processors = torch.nn.ModuleDict()

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(input_features, attention_mask)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # Processors are torchaudio transforms recreated on demand; strip their saved
        # buffers so load_from_checkpoint works regardless of which device trained the model.
        to_remove = [k for k in checkpoint["state_dict"] if k.startswith("processors.")]
        for k in to_remove:
            del checkpoint["state_dict"][k]

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        # The raw batch contains frozen dataclasses (RuntimeExample) that Lightning cannot
        # move to device via setattr. Device placement is handled inside _process_raw_batch.
        return batch

    def training_step(self, raw_batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch_size = len(raw_batch["examples"])
        batch = self._process_raw_batch(raw_batch)
        outputs = self(batch["input_features"], batch["attention_mask"])
        loss, metrics = self._loss(outputs, batch)
        self.log_dict({f"train/{key}": value for key, value in metrics.items()}, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, raw_batch: dict[str, Any], batch_idx: int) -> None:
        batch_size = len(raw_batch["examples"])
        batch = self._process_raw_batch(raw_batch)
        outputs = self(batch["input_features"], batch["attention_mask"])
        _, metrics = self._loss(outputs, batch)
        self.log_dict({f"val/{key}": value for key, value in metrics.items()}, prog_bar=True, on_epoch=True, batch_size=batch_size)

    def test_step(self, raw_batch: dict[str, Any], batch_idx: int) -> None:
        batch_size = len(raw_batch["examples"])
        batch = self._process_raw_batch(raw_batch)
        outputs = self(batch["input_features"], batch["attention_mask"])
        _, metrics = self._loss(outputs, batch)
        self.log_dict({f"test/{key}": value for key, value in metrics.items()}, prog_bar=True, on_epoch=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config_obj.learning_rate,
            weight_decay=self.config_obj.weight_decay,
        )
        return optimizer

    def process_raw_batch(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        return self._process_raw_batch(raw_batch)

    def _process_raw_batch(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        key = device_key(self.device)
        if key not in self.processors:
            max_samples = None
            if self.config_obj.max_audio_sec is not None:
                max_samples = int(self.config_obj.max_audio_sec * self.config_obj.features.sample_rate)
            self.processors[key] = OnsetOffsetBatchProcessor(
                feature_config=self.config_obj.features,
                label_config=self.config_obj.labels,
                device=self.device,
                use_amp_features=self.device.type == "cuda",
                max_audio_samples=max_samples,
            )
        return self.processors[key](raw_batch)

    def _loss(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = batch["attention_mask"].float()
        labels = batch["labels"]
        onset_loss = masked_bce_with_logits(
            outputs["onset_logits"],
            labels["onset"],
            mask,
            self.config_obj.onset_pos_weight,
        )
        offset_loss = masked_bce_with_logits(
            outputs["offset_logits"],
            labels["offset"],
            mask,
            self.config_obj.offset_pos_weight,
        )
        active_loss = masked_bce_with_logits(
            outputs["active_logits"],
            labels["active"],
            mask,
            self.config_obj.active_pos_weight,
        )
        loss = onset_loss + offset_loss + active_loss
        with torch.no_grad():
            onset_prob = torch.sigmoid(outputs["onset_logits"])
            offset_prob = torch.sigmoid(outputs["offset_logits"])
            active_prob = torch.sigmoid(outputs["active_logits"])
            metrics = {
                "loss": loss.detach(),
                "onset_loss": onset_loss.detach(),
                "offset_loss": offset_loss.detach(),
                "active_loss": active_loss.detach(),
                "onset_peak": masked_mean(onset_prob, mask),
                "offset_peak": masked_mean(offset_prob, mask),
                "active_mean": masked_mean(active_prob, mask),
            }
        return loss, metrics


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: float) -> torch.Tensor:
    positive_weight = torch.tensor(pos_weight, dtype=targets.dtype, device=targets.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=positive_weight)
    loss = loss * mask
    return loss.sum() / torch.clamp(mask.sum(), min=1.0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def device_key(device: torch.device) -> str:
    return str(device).replace(":", "_")
