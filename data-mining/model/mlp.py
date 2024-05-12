import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassAUROC


class SimpleMLPModel(L.LightningModule):
    def __init__(
        self,
        in_features: int = 27,
        out_features: int = 7,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.num_classes = out_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            torch.nn.ReLU(),
            nn.Linear(128, 64),
            torch.nn.ReLU(),
            nn.Linear(64, out_features),
        )
        self.head = torch.nn.Softmax(dim=1)

        self.net = nn.Sequential(self.mlp, self.head)

        self.val_step_outputs = []

        self.auc = MulticlassAUROC(num_classes=self.num_classes, average="macro", thresholds=None)

        self.init_weight()
        self.save_hyperparameters()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.net(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.net(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("val/loss", loss, prog_bar=True)

        self.val_step_outputs.append({"pred": x_hat, "target": y})
        return loss

    def on_validation_epoch_end(self) -> None:
        if len(self.val_step_outputs) == 0:
            return
        preds = torch.cat([x["pred"] for x in self.val_step_outputs])
        targets = torch.cat([x["target"].argmax(dim=1) for x in self.val_step_outputs])
        self.log("val/auc", self.auc(preds, targets), prog_bar=True)
        self.val_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x = self.net(x[:, 1:])
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        return optimizer
