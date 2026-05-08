from pathlib import Path
import pytorch_lightning as pl

from parking_spaces_classifier.datasets.datamodule import PKLotDataModule, DatasetConfig
from parking_spaces_classifier.models.baseline_model import get_baseline_detector
from parking_spaces_classifier.models.detector_model import BaseDetectorLitModule


def train():
    train_config = DatasetConfig(
        root_dir=Path("data/raw/train"),
        annotations=Path("data/raw/train/_annotations.coco.json")
    )
    
    val_config = DatasetConfig(
        root_dir=Path("data/raw/valid"), 
        annotations=Path("data/raw/valid/_annotations.coco.json")
    )

    datamodule = PKLotDataModule(
        train=train_config,
        val=val_config,
        batch_size=8,  
        num_workers=2
    )

    pytorch_model = get_baseline_detector(num_classes=3, freeze_all_but_head=True)

    lightning_model = BaseDetectorLitModule(model=pytorch_model, lr=1e-4)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto", 
        devices=1,
        log_every_n_steps=5,
        limit_train_batches=50,
        accumulate_grad_batches=8,  
        limit_val_batches=5      
    )

    print("Inicjalizacja zakończona. Rozpoczynam fit...")
    trainer.fit(model=lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    train()
