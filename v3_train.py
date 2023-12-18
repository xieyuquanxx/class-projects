from argparse import ArgumentParser

import lightning as L
import torch
import torch.utils.data as data
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from loguru import logger
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader
from transformers import logging

from datasets.final_dataset import ELDataset
from model.v3_model import BertELModelV3

if __name__ == "__main__":
    logging.set_verbosity_error()
    # 要用wandb,需要挂梯子
    # os.environ["http_proxy"] = "http://127.0.0.1:7890"
    # os.environ["https_proxy"] = "http://127.0.0.1:7890"

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=103)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-chinese",
        choices=["chinese-roberta-wwm-ext", "bert-base-chinese", "ernie-3.0-base-zh"],
    )

    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    train_dataset = ELDataset("train", max_length=args.max_length)
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size

    # split the train set into two
    train_set, valid_set = data.random_split(
        train_dataset,
        [train_set_size, valid_set_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = BertELModelV3(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZjA5MmE1My05NDUxLTQ4YWItOGRlOS1lY2E1MzRmMjY1MmUifQ==",
        project="xieyuquanxx/NLP-EL",
    )
    logger.info("Load Model... Start Training...")
    trainer = L.Trainer(
        limit_train_batches=0.9,
        val_check_interval=0.5,
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir=f"logs/{args.model_name}",
        logger=neptune_logger,  # type: ignore
        callbacks=[
            ModelCheckpoint(
                dirpath="check_points/bert-chi",
                save_top_k=-1,
                every_n_epochs=1,
                monitor="val/loss",
                filename="bert-chi{epoch}",
            ),
            RichProgressBar(),
        ],
    )
    neptune_logger.log_model_summary(model=model, max_depth=-1)  # type: ignore

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path="/shao_rui/xyq/nlp_el/demos/check_points/new-net-roberta/roberta-wwm-epoch=5.ckpt",
    )
