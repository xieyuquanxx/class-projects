import hydra
import hydra.core.hydra_config
import lightning
import pandas as pd
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from omegaconf import DictConfig

from data import SteelPlateDataModule
from model import SimpleMLPModel


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    lightning.seed_everything(cfg["trainer"]["seed"])

    data_module = SteelPlateDataModule(cfg["data"]["path"], cfg["data"]["batch_size"])
    model = SimpleMLPModel.load_from_checkpoint(cfg["model"]["ckpt"])
    model.eval()

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg["trainer"]["gpus"],
        callbacks=[RichProgressBar()],
    )
    result = trainer.predict(model, datamodule=data_module)
    assert result, "Prediction failed"

    labels = ["id", "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

    df = pd.DataFrame(columns=labels)
    start_id = 19219
    for i, r in enumerate(result):
        df.loc[i] = [start_id + i] + r[0].tolist()

    df.to_csv("outputs/submission/submission.csv", index=False)


if __name__ == "__main__":
    main()
