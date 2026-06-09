from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .custom_detector import decode_predictions, focal_loss, reg_l1_loss


class BaseDetectorLitModule(pl.LightningModule):
    def __init__(self, model: Module, lr: float = 1e-4) -> None:
        super().__init__()
        self.lr = lr
        self.model = model

    def forward(self, images: list[Tensor], targets: list[dict[str, Tensor]] | None = None):
        return self.model(images, targets)

    def _shared_step(self, batch, stage: str) -> Tensor:
        images, targets = batch
        loss_dict = self.model(list(images), list(targets))
        loss = sum(loss_dict.values())
        self.log(f"{stage}/loss", loss, prog_bar=True)
        for name, value in loss_dict.items():
            self.log(f"{stage}/{name}", value, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class CenterNetLitModule(pl.LightningModule):
    def __init__(self, model: Module, lr: float = 3e-4, epochs: int = 25) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        # Inicjalizacja metryk
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy", class_metrics=True, max_detection_thresholds=[10, 100, 1000]
        )

    def forward(self, images: Tensor):
        return self.model(images)

    def _shared_step(self, batch, stage: str):
        images, targets = batch

        # Pytorch Lightning przekazuje listę tensorów w batchu jeśli użyjemy własnego collate_fn.
        # Images to krotka/lista tensorów (C, H, W). Sklejamy ją:
        images_tensor = torch.stack(list(images)).to(self.device)
        B, C, H, W = images_tensor.shape
        gt_hm = torch.stack([t["heatmap"] for t in targets]).to(self.device)
        gt_wh = torch.stack([t["wh"] for t in targets]).to(self.device)
        gt_offset = torch.stack([t["offset"] for t in targets]).to(self.device)
        reg_mask = torch.stack([t["reg_mask"] for t in targets]).to(self.device)

        preds = self(images_tensor)
        pred_hm = preds["heatmap"]
        pred_wh = preds["wh"]
        pred_offset = preds["offset"]

        hm_loss = focal_loss(pred_hm, gt_hm)
        wh_loss = reg_l1_loss(pred_wh, gt_wh, reg_mask)
        offset_loss = reg_l1_loss(pred_offset, gt_offset, reg_mask)

        total_loss = hm_loss + 0.1 * wh_loss + 1.0 * offset_loss

        self.log(f"{stage}/loss", total_loss, prog_bar=True, batch_size=B)
        self.log(f"{stage}/hm_loss", hm_loss, prog_bar=False, batch_size=B)
        self.log(f"{stage}/wh_loss", wh_loss, prog_bar=False, batch_size=B)
        self.log(f"{stage}/offset_loss", offset_loss, prog_bar=False, batch_size=B)

        # Zbieranie predykcji do metryk podczas walidacji / testowania
        if stage in ["val", "test"]:
            decoded_preds = decode_predictions(pred_hm, pred_wh, pred_offset, threshold=0.3)

            fixed_targets = []
            for t in targets:
                t_dict = {}
                for k, v in t.items():
                    if k == "boxes" and v.numel() == 0:
                        t_dict[k] = torch.empty((0, 4), dtype=torch.float32, device=self.device)
                    else:
                        t_dict[k] = v.to(self.device)
                fixed_targets.append(t_dict)

            self.map_metric.update(decoded_preds, fixed_targets)

        return total_loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        results = self.map_metric.compute()
        self.log("val/mAP_50", results["map_50"], prog_bar=True)
        self.log("val/mAP_75", results["map_75"], prog_bar=False)
        self.log("val/mAP_50-95", results["map"], prog_bar=True)
        self.map_metric.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        results = self.map_metric.compute()
        self.log("test/mAP_50", results["map_50"])
        self.log("test/mAP_75", results["map_75"])
        self.log("test/mAP_50-95", results["map"])
        self.map_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=0.0005,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return [optimizer], [scheduler]


def load_centernet_checkpoint(
    checkpoint_path: str | Path,
    model: Module | None = None,
    *,
    num_classes: int = 2,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CenterNetLitModule:
    """Load a trained CenterNet Lightning module from a checkpoint.

    If no base model is provided, the default SimpleUNetCenterNet backbone is created.
    This keeps notebook usage short while still allowing custom architectures.
    """

    if model is None:
        from .custom_detector import SimpleUNetCenterNet

        model = SimpleUNetCenterNet(num_classes=num_classes)

    lit_model = CenterNetLitModule(model=model)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    lit_model.load_state_dict(state_dict, strict=strict)
    return lit_model
