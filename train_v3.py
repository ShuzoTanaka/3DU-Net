import torch
from pathlib import Path
import datetime
import lightning as L
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from DataModuleSafe import DataModuleSafe
from model import MultiClassModel

# 安全な34症例（ホールドアウト完全除外）で学習
# Dice + CE loss、seed=42固定分割
if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    base_dir = Path(__file__).resolve().parent
    old_data_path = base_dir / "0206data" / "train_val"
    new_data_path = base_dir / "Dataset001_lumber"

    data_module = DataModuleSafe(
        old_data_path=old_data_path,
        new_data_path=new_data_path,
        batch_size=2,
    )

    model = MultiClassModel(
        in_channels=1,
        num_classes=3,
        encoder_name="efficientnet-b0",
        lr=1e-3,
    )

    dt = datetime.datetime.now()
    logger = TensorBoardLogger(
        "logs",
        name=dt.strftime("%Y-%m-%d_%H-%M-%S"),
        version="version_0",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="3d_v3",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=200,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    for images, masks in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch mask shape: {masks.shape}")
        break

    print("Is instance of LightningModule:", isinstance(model, pl.LightningModule))
    trainer.fit(model, datamodule=data_module)
