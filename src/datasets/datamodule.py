from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import PKLotDataset, collate_fn, get_transform


@dataclass
class DatasetConfig:
    root_dir: Path
    annotations: Path


class PKLotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: DatasetConfig,
        val: DatasetConfig,
        test: DatasetConfig | None = None,
        batch_size: int = 4,
        num_workers: int = 2,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        train_subset_size: int | float | None = None,
        val_subset_size: int | float | None = None,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.train_config = train
        self.val_config = val
        self.test_config = test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size
        self.persistent_workers = persistent_workers
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def _get_subset_size(self, dataset: Dataset, subset_size: int | float | None) -> int | None:
        if subset_size is None:
            return None
        if isinstance(subset_size, float):
            if not (0.0 < subset_size <= 1.0):
                raise ValueError(f"Fractional subset size must be between 0.0 and 1.0, got {subset_size}")
            return int(len(dataset) * subset_size)
        return subset_size

    def setup(self, stage: str | None = None) -> None:
        if self.train_transforms is None:
            self.train_transforms = get_transform(is_train=True)
        if self.val_transforms is None:
            self.val_transforms = get_transform(is_train=False)
        if self.test_transforms is None:
            self.test_transforms = get_transform(is_train=False)


        match stage:
            case "fit" | None:
                train_ds = PKLotDataset(
                    root_dir=self.train_config.root_dir,
                    annotation_file=self.train_config.annotations,
                    transforms=self.train_transforms,
                )
                train_size = self._get_subset_size(train_ds, self.train_subset_size)
                self.train_dataset = Subset(train_ds, range(train_size)) if train_size else train_ds

                val_ds = PKLotDataset(
                    root_dir=self.val_config.root_dir,
                    annotation_file=self.val_config.annotations,
                    transforms=self.val_transforms,
                )
                val_size = self._get_subset_size(val_ds, self.val_subset_size)
                self.val_dataset = Subset(val_ds, range(val_size)) if val_size else val_ds
            case "validate":
                val_ds = PKLotDataset(
                    root_dir=self.val_config.root_dir,
                    annotation_file=self.val_config.annotations,
                    transforms=self.val_transforms,
                )
                val_size = self._get_subset_size(val_ds, self.val_subset_size)
                self.val_dataset = Subset(val_ds, range(val_size)) if val_size else val_ds
            case "test":
                if self.test_config is None:
                    raise RuntimeError("Test dataset config is not provided.")
                self.test_dataset = PKLotDataset(
                    root_dir=self.test_config.root_dir,
                    annotation_file=self.test_config.annotations,
                    transforms=self.test_transforms,
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
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
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
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not initialized. Call setup().")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn,
        )
