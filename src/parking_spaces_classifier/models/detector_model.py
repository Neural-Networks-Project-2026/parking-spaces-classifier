from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import Module


class BaseDetectorLitModule(pl.LightningModule):
    def __init__(self, model: Module, lr: float = 1e-4) -> None:
        super().__init__()
        self.lr = lr
        self.model = model

    def forward(self, images: list[Tensor], targets: list[dict[str, Tensor]] | None = None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        images, targets = batch

        loss_dict = self.model(list(images), list(targets))
        loss = sum(loss_dict.values())
        
        self.log("train/loss", loss, prog_bar=True)
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, prog_bar=False)
            
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        images, targets = batch

        preds = self.model(list(images))
        
        # W przyszłości: liczenie metryki mAP z torchmetrics.detection
        # self.map_metric(preds, targets)

    def test_step(self, batch, batch_idx: int) -> None:
        images, targets = batch
        preds = self.model(list(images))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=1e-4)
