from argparse import ArgumentParser

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from loguru import logger
from torch.utils.data import DataLoader
from transformers import logging

from datasets.final_dataset import ELDataset
from model.v2_bert_model import BertELModelV2

if __name__ == "__main__":
    logging.set_verbosity_error()

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=103)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name",
        type=str,
        default="chinese-roberta-wwm-ext",
        choices=["chinese-roberta-wwm-ext", "bert-base-chinese", "ernie-3.0-base-zh"],
    )

    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    # wandb_logger = WandbLogger(project="NLP-EL", log_model="all")

    train_dataset = ELDataset("train", max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = BertELModelV2(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    logger.info("Load Model... Start Training...")
    trainer = L.Trainer(
        limit_train_batches=0.9,
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir="ckpt/",
        # logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath="check_points/new-net-roberta/",
                save_top_k=-1,
                every_n_epochs=1,
                filename="roberta-wwm-{epoch}",
            ),
            RichProgressBar(),
        ],
    )
    # wandb_logger.watch(model)

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        # ckpt_path="/shao_rui/xyq/nlp_el/demos/check_points/roberta-wwm/roberta-wwm-epoch=1.ckpt",
    )
    # wandb.finish()
