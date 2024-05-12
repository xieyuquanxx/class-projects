import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class SimpleLinearModel(L.LightningModule):
    def __init__(self, in_features=27, out_features=7, lr=1e-4):
        super().__init__()

        self.lr = lr

        self.linear = nn.Linear(in_features, out_features)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.linear(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.linear(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
