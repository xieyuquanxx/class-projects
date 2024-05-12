import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import BertModel, BertTokenizer, ErnieModel


class BertELModelV3(L.LightningModule):
    def __init__(
        self,
        model_name="bert-base-chinese",
        max_length=128,
        batch_size=256,
        temperature=0.1,
        lr=5e-5,
        weight_decay=1e-7,
    ):
        super(BertELModelV3, self).__init__()
        # bert模型
        if "ernie" in model_name:
            self.bert = ErnieModel.from_pretrained("ernie-3.0-base-zh")
        else:
            self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # 定义bert后面要接的网络

        self.sentence_net = nn.Sequential(
            nn.Linear(768 * 3, 128),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

        self.nil_type_net = nn.Sequential(
            nn.Linear(768 * 3, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 24),
        )

        self.max_length = max_length
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.version = "v3, more deep network"

        self.init_weight()

        self.save_hyperparameters()

    def init_weight(self):
        """
        Initializes the weights of the linear layers in the model.

        This method iterates through all the modules in the model and checks if a module is a linear layer.
        If a module is a linear layer, it initializes the weight with a normal distribution (mean=0, std=0.02)
        and sets the bias to a constant value of 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, y_match, y_type, loc = batch
        x_match, x_type = self(input_ids, attention_mask, loc)

        loss1 = F.binary_cross_entropy(x_match, y_match.to(torch.float32))
        loss2 = F.cross_entropy(x_type / self.temperature, y_type)
        # Logging to TensorBoard (if installed) by default
        loss = loss1 + loss2
        self.log("train/match_loss", loss1)
        self.log("train/type_loss", loss2)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, y_match, y_type, loc = batch
        x_match, x_type = self(input_ids, attention_mask, loc)

        loss1 = F.binary_cross_entropy(x_match, y_match.to(torch.float32))
        loss2 = F.cross_entropy(x_type / self.temperature, y_type)
        # Logging to TensorBoard (if installed) by default
        loss = loss1 + loss2
        self.log("val/match_loss", loss1, sync_dist=True)
        self.log("val/type_loss", loss2, sync_dist=True)
        self.log("val/loss", loss, sync_dist=True)

    def get_concat_clstoken(self, outputs, loc):
        # # 是108，拼接的向量是 CLS, #实体#的开始和结束token
        # 获取bert输出的隐藏层特征

        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]
        all_tokens = []
        for bs in range(0, self.batch_size):
            if bs >= len(loc[0]) or bs >= len(loc[1]):
                break
            tt1 = last_hidden_state[bs, loc[0][bs], :]
            tt2 = last_hidden_state[bs, loc[1][bs], :]
            t = torch.cat([tt1, tt2], dim=0)
            # print(t.shape)
            all_tokens.append(t)
            # break
        # t1 = last_hidden_state[:, loc[0], :]  # [batch_size, 768]
        # t2 = last_hidden_state[:, loc[1], :]  # [batch_size, 768]
        # print(loc)
        all_tokens = torch.stack(tuple(all_tokens), dim=0)
        # print(all_tokens.shape)  # [batch_size, 768*2]
        cls_token = outputs.pooler_output  # [batch_size, 768]
        t = torch.cat([cls_token, all_tokens], dim=1)  # [batch_size, 768*3]
        # print(t.shape)
        return t

    def forward(self, input_ids, attention_masks, loc):
        outputs = self.bert(input_ids, attention_mask=attention_masks)  # type: ignore
        t = self.get_concat_clstoken(outputs, loc)  # [batch_size, 768*3]

        match_tensor = self.sentence_net(t)  # [batch_size, 1]
        type_tensor = self.nil_type_net(t)  # [batch_size, 24]

        match_tensor = torch.sigmoid(match_tensor)
        match_tensor = match_tensor.squeeze(dim=1)
        # print(out_match, out_match.shape)
        # out_type = F.softmax(type_tensor, dim=1)  # [batch_size,24]
        # 这里不能softmax，因为F.cross_entropy已经包含了softmax
        # out_type = type_tensor / self.temperature
        return match_tensor, type_tensor

    def eval_get_clstoken(self, texta, textb, textc=[]):
        if len(textb) == 0:
            train_encodings = self.tokenizer(
                texta,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
        else:
            if len(textc) == 0:
                train_encodings = self.tokenizer(
                    texta,
                    textb,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
            else:
                train_encodings = self.tokenizer(
                    texta,
                    textb,
                    textc,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
        input_list = train_encodings["input_ids"]
        # print(input_list)
        first_index = input_list[0].index(108)
        second_index = input_list[0].index(108, first_index + 1)

        outputs = self.bert(
            torch.tensor(input_list).to("cuda"),
            torch.tensor(train_encodings["attention_mask"]).to("cuda"),
        )  # type: ignore

        t = self.get_concat_clstoken(
            outputs, [[first_index + 1], [second_index - 1]]
        )  # [1, 768*3]
        return t

    def predict(self, texta, textb, textc=[]):
        """
        匹配两个句子
        :param texta:
        :param textb:
        :return:
        """
        if len(textc) == 0:
            t = self.eval_get_clstoken(texta, textb)
        else:
            t = self.eval_get_clstoken(texta, textb, textc)  # [1, 768*3]
        x_match = self.sentence_net(t).squeeze(0)  # [1]
        return torch.sigmoid(x_match)

    def get_type(self, texta):
        t = self.eval_get_clstoken(texta, [])
        x_type = self.nil_type_net(t)
        out_type = torch.argmax(F.softmax(x_type, dim=1))  # [1,24]

        return out_type
