import json

import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer


class ELDataset(Dataset):
    def __init__(self, mode="train", max_length=128) -> None:
        super().__init__()
        self.mode = mode
        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.data = self._load_data()

    def _load_data(self):
        logger.info("Loading data...")

        with open("data/merged_v2.json", "r", encoding="utf8") as f:
            data = [json.loads(line) for line in f.readlines()]
        # data = data[:300]
        # load data need 10m
        for d in tqdm(data, total=len(data)):
            d["text"] = self.tokenizer(
                d["texta"],
                d["textb"],
                d["textc"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

        logger.info(f"{self.mode} data loaded.")
        return data

    def __getitem__(self, index):
        sample = self.data[index]
        input_ids = sample["text"]["input_ids"]
        # print(input_ids)
        try:
            first_index = input_ids.index(108)
        except Exception:
            first_index = 0
        try:
            second_index = input_ids.index(108, first_index + 1)
        except Exception:
            second_index = 3
        # print(sample)
        return (
            torch.tensor(input_ids),
            torch.tensor(sample["text"]["attention_mask"]),
            torch.tensor(sample["label"], dtype=torch.int64),
            torch.tensor(sample["target"], dtype=torch.int64),
            (first_index + 1, second_index - 1),
        )

    def __len__(self):
        return len(self.data)
