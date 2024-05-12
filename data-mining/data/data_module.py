import os

import lightning as L  # noqa: N812
from torch.utils.data import DataLoader

from .dataset import SteelPlateDataset


class SteelPlateDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 32):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train_dataset = SteelPlateDataset(os.path.join(self.data_dir, "train_data.csv"), train=True)
        self.val_dataset = SteelPlateDataset(os.path.join(self.data_dir, "val_data.csv"), train=True)

        self.test_dataset = SteelPlateDataset(os.path.join(self.data_dir, "test.csv"), train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)
