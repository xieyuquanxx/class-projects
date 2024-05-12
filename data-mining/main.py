import logging

import hydra
import hydra.core.hydra_config
import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from data import SteelPlateDataModule
from model import SimpleMLPModel

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    wandb_logger = WandbLogger(
        project="2024Spring-data-minnig",
        save_dir="outputs",
        offline=True,
    )
    lightning.seed_everything(cfg["trainer"]["seed"])
    logger.info(f"Seed set to {cfg['trainer']['seed']}")

    data_module = SteelPlateDataModule(cfg["data"]["path"], cfg["data"]["batch_size"])
    logger.info(f"DataModule created with batch size {cfg['data']['batch_size']}")

    model = SimpleMLPModel(
        cfg["model"]["in_features"],
        cfg["model"]["num_classes"],
        lr=cfg["model"]["lr"],
        momentum=cfg["model"]["momentum"],
        weight_decay=cfg["model"]["weight_decay"],
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg["trainer"]["gpus"],
        max_epochs=cfg["trainer"]["max_epochs"],
        min_epochs=cfg["trainer"]["min_epochs"],
        logger=wandb_logger,
        callbacks=[RichProgressBar(), EarlyStopping(monitor="val/loss", mode="min")],
    )

    trainer.fit(
        model,
        datamodule=data_module,
    )
    logger.info("Training completed")


if __name__ == "__main__":
    main()
