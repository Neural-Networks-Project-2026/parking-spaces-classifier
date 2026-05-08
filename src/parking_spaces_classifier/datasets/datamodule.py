from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from parking_spaces_classifier.datasets.dataset import PKLotDataset, collate_fn


@dataclass
class DatasetConfig:
    root_dir: Path
    annotations: Path


class PKLotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: DatasetConfig,
        val: DatasetConfig,
        batch_size: int = 4,
        num_workers: int = 2,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
    ) -> None:
        super().__init__()
        self.train_config = train
        self.val_config = val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.train_dataset: PKLotDataset | None = None
        self.val_dataset: PKLotDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        match stage:
            case "fit" | None:
                self.train_dataset = PKLotDataset(
                    root_dir=self.train_config.root_dir,
                    annotation_file=self.train_config.annotations,
                    transforms=self.train_transforms,
                )
                self.val_dataset = PKLotDataset(
                    root_dir=self.val_config.root_dir,
                    annotation_file=self.val_config.annotations,
                    transforms=self.val_transforms,
                )
            case "validate":
                self.val_dataset = PKLotDataset(
                    root_dir=self.val_config.root_dir,
                    annotation_file=self.val_config.annotations,
                    transforms=self.val_transforms,
                )
            case _:
                raise ValueError(f"Unsupported stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not initialized. Call setup().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not initialized. Call setup().")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
