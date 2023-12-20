from argparse import ArgumentParser

import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from loguru import logger
from torch.utils.data import DataLoader
from transformers import logging

from datasets.final_dataset import ELDataset
from model.v4_model import BertELModelV4

if __name__ == "__main__":
    logging.set_verbosity_error()

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=103)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-chinese",
        choices=["chinese-roberta-wwm-ext", "bert-base-chinese"],
    )

    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    train_dataset = ELDataset(
        "train", max_length=args.max_length, model_name=args.model_name
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = BertELModelV4(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=0,
    )

    logger.info("Load Model... Start Training...")
    trainer = L.Trainer(
        limit_train_batches=0.9,
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir=f"{args.model_name}",
        callbacks=[
            ModelCheckpoint(
                dirpath="check_points/bertv4",
                save_top_k=-1,
                every_n_epochs=1,
                monitor="train/loss",
                filename="bertv4{epoch}",
            ),
            RichProgressBar(),
        ],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
    )
