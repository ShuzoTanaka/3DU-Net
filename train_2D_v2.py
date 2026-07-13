import torch
import datetime
import lightning as L
import pytorch_lightning as pl
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from DataModule2DSafe import DataModule2DSafe
from model_2D import MultiClassModel

# 2D U-Net: 34症例の全スライス使用（最大データ量）
# Dice loss only、EfficientNet-B0、batch_size=16（2Dなので大きめ）
if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    base_dir = Path(__file__).resolve().parent
    old_data_path = base_dir / "0206data" / "train_val"
    new_data_path = base_dir / "Dataset001_lumber"

    data_module = DataModule2DSafe(
        old_data_path=old_data_path,
        new_data_path=new_data_path,
        batch_size=16,
        seed=None,
    )

    model = MultiClassModel(
        in_channels=1,
        num_classes=3,
        encoder_name="efficientnet-b0",
    )

    dt = datetime.datetime.now()
    logger = TensorBoardLogger(
        "logs",
        name=dt.strftime("%Y-%m-%d_%H-%M-%S"),
        version="version_0",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="3d_2D_v2",
        filename="best-2D-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=100,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
    )

    print("Is LightningModule:", isinstance(model, pl.LightningModule))
    print("Training: 2D U-Net, all slices, 34 safe cases, Dice loss only")

    data_module.setup()
    trainer.fit(model, datamodule=data_module)
