import lightning as L
import torch
import torch.nn.functional as F

# from pytorch_lightning.loggers import WandbLogger
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, logging

# from model_eval import eeval
from dataset.final_dataset import ELDataset

# import wandb
# import os


class BertELModel(L.LightningModule):
    def __init__(
        self,
        model_name="bert-base-chinese",
        max_length=128,
        temperature=0.1,
        lr=5e-5,
        weight_decay=1e-7,
    ):
        super(BertELModel, self).__init__()
        # bert模型
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # 定义bert后面要接的网络
        # self.nil_type_ffn = nn.Linear(768, 24)  # type 24分类
        # self.sentence_ffn = nn.Linear(768, 2)  # 文本匹配 2分类

        self.sentence_net = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(768 * 3, 128), nn.Dropout(0.1), nn.Linear(128, 1)
        )

        self.nil_type_net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 * 3, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 24),
        )

        self.max_length = max_length
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        input_ids, attention_mask, y_match, y_type = batch
        # x = self(input_ids, attention_mask)
        x_match, x_type = self(input_ids, attention_mask)

        loss1 = F.binary_cross_entropy(x_match, y_match.to(torch.float32))
        loss2 = F.cross_entropy(x_type, y_type)
        # Logging to TensorBoard (if installed) by default
        loss = loss1 + loss2
        self.log("train_match_loss", loss1)
        self.log("train_type_loss", loss2)
        self.log("train_loss", loss)
        return loss

    def get_concat_clstoken(self, outputs):
        # # 是108，拼接的向量是 CLS, #实体#的开始和结束token
        # 获取bert输出的隐藏层特征
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]
        t1 = last_hidden_state[:, 1, :]  # [batch_size, 768]
        t2 = last_hidden_state[:, 2, :]  # [batch_size, 768]
        cls_token = outputs.pooler_output  # [batch_size, 768]
        t = torch.cat([cls_token, t1, t2], dim=1)  # [batch_size, 768*3]
        return t

    def forward(self, input_ids, attention_masks):
        # 输入bert
        outputs = self.bert(input_ids, attention_mask=attention_masks)  # type: ignore
        t = self.get_concat_clstoken(outputs)  # [batch_size, 768*3]
        # logger.info("cat tensor T: {}", t.shape)

        match_tensor = self.sentence_net(t)  # [batch_size, 1]
        type_tensor = self.nil_type_net(t)  # [batch_size, 24]

        # out_match = F.softmax(match_tensor, dim=1)
        # out_match = out_match / self.temperature
        match_tensor = torch.sigmoid(match_tensor)
        match_tensor = match_tensor.squeeze(dim=1)
        # print(out_match, out_match.shape)
        out_type = F.softmax(type_tensor, dim=1)  # [batch_size,24]
        out_type = out_type / self.temperature
        # print(out.shape)
        # print(out.item())
        return match_tensor, out_type

    def predict(self, texta, textb):
        """
        匹配两个句子
        :param texta:
        :param textb:
        :return:
        """
        # print(input_ids)
        # print(attention_mask)
        train_encodings = self.tokenizer(
            texta,
            textb,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        outputs = self.bert(
            torch.tensor(train_encodings["input_ids"]).to("cuda"),
            torch.tensor(train_encodings["attention_mask"]).to("cuda"),
        )  # type: ignore
        t = self.get_concat_clstoken(outputs)  # [batch_size, 768*3]
        # cls_token = outputs.pooler_output
        x_match = self.sentence_net(t).squeeze(0)  # [1]
        return x_match
        # x_match = F.softmax(x_match, dim=0)
        # id = torch.argmax(x_match)
        # if id == 0:  # 最大的是不相关,直接认为是NIL
        #     return torch.tensor(0).to("cuda")
        # return x_match[id]

    def get_type(self, texta):
        train_encodings = self.tokenizer(
            texta,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        outputs = self.bert(
            torch.tensor(train_encodings["input_ids"]).to("cuda"),
            torch.tensor(train_encodings["attention_mask"]).to("cuda"),
        )  # type: ignore
        t = self.get_concat_clstoken(outputs)  # [batch_size, 768*3]
        x_type = self.nil_type_net(t)
        out_type = torch.argmax(F.softmax(x_type, dim=1))  # [batch_size,24]

        return out_type


if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "offline"
    logging.set_verbosity_error()
    L.seed_everything(42, workers=True)

    train_dataset = ELDataset("train", max_length=128)
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=256, num_workers=103
    )

    # wandb_logger = WandbLogger(project="my-nlp-el")
    # wandb_logger.experiment.config["batch_size"] = 128

    model = BertELModel(max_length=128)
    # model = BertELModel.load_from_checkpoint(
    #     "/shao_rui/xyq/nlp_el/demos/ckpt/lightning_logs/version_3/checkpoints/epoch=6-step=9338.ckpt"
    # )
    logger.info("Load Model... Start Training...")
    trainer = L.Trainer(
        limit_train_batches=0.7,
        max_epochs=10,
        accelerator="auto",
        default_root_dir="ckpt/",
        # fast_dev_run=True,
        # logger=wandb_logger,  # type: ignore
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
