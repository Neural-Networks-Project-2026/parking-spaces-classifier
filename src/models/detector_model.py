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
