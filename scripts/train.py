import os
import sys

# Dodajemy folder główny do PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch_lightning.cli import LightningCLI

from src.models.lit_module import CenterNetLitModule
from src.datasets.datamodule import PKLotDataModule

def cli_main():
    cli = LightningCLI(
        model_class=CenterNetLitModule,
        datamodule_class=PKLotDataModule,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == "__main__":
    cli_main()
