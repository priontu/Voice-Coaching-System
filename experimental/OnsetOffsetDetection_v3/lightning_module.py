from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

try:
    import lightning as L
except ImportError as exc:
    raise RuntimeError("lightning is required for OnsetOffsetDetection_v3") from exc

from OnsetOffsetDetection_v1.lightning_module import device_key, masked_bce_with_logits, masked_mean
from OnsetOffsetDetection_v1.preprocessing import OnsetOffsetBatchProcessor, OnsetOffsetFeatureConfig, OnsetOffsetLabelConfig

from .model import CoupledModelConfig, OnsetOffsetCoupledConformer


@dataclass
class OnsetOffsetTrainingConfig:
    model: CoupledModelConfig = field(default_factory=CoupledModelConfig)
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
        self.model = OnsetOffsetCoupledConformer(config.model)
        self.processors = torch.nn.ModuleDict()

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(input_features, attention_mask)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        to_remove = [k for k in checkpoint["state_dict"] if k.startswith("processors.")]
        for k in to_remove:
            del checkpoint["state_dict"][k]

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        return batch

    def training_step(self, raw_batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._process_raw_batch(raw_batch)
        outputs = self(batch["input_features"], batch["attention_mask"])
        loss, metrics = self._loss(outputs, batch)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(raw_batch["examples"]))
        return loss

    def validation_step(self, raw_batch: dict[str, Any], batch_idx: int) -> None:
        batch = self._process_raw_batch(raw_batch)
        outputs = self(batch["input_features"], batch["attention_mask"])
        _, metrics = self._loss(outputs, batch)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True, on_epoch=True, batch_size=len(raw_batch["examples"]))

    def test_step(self, raw_batch: dict[str, Any], batch_idx: int) -> None:
        batch = self._process_raw_batch(raw_batch)
        outputs = self(batch["input_features"], batch["attention_mask"])
        _, metrics = self._loss(outputs, batch)
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, prog_bar=True, on_epoch=True, batch_size=len(raw_batch["examples"]))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config_obj.learning_rate, weight_decay=self.config_obj.weight_decay)

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

    def _loss(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        mask = batch["attention_mask"].float()
        labels = batch["labels"]
        onset_loss = masked_bce_with_logits(outputs["onset_logits"], labels["onset"], mask, self.config_obj.onset_pos_weight)
        offset_loss = masked_bce_with_logits(outputs["offset_logits"], labels["offset"], mask, self.config_obj.offset_pos_weight)
        active_loss = masked_bce_with_logits(outputs["active_logits"], labels["active"], mask, self.config_obj.active_pos_weight)
        loss = onset_loss + offset_loss + active_loss
        with torch.no_grad():
            metrics = {
                "loss": loss.detach(),
                "onset_loss": onset_loss.detach(),
                "offset_loss": offset_loss.detach(),
                "active_loss": active_loss.detach(),
                "onset_peak": masked_mean(torch.sigmoid(outputs["onset_logits"]), mask),
                "offset_peak": masked_mean(torch.sigmoid(outputs["offset_logits"]), mask),
                "active_mean": masked_mean(torch.sigmoid(outputs["active_logits"]), mask),
            }
        return loss, metrics
