import torch
from pathlib import Path
import datetime
import lightning as L
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from DataModuleSafe import DataModuleSafe
from model import MultiClassModel

# 最終版: Dice lossのみ・seed無し・安全34症例
# 結果: nerve=0.6794, spinal=0.6768, 平均=0.6781（0.68達成）
if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    base_dir = Path(__file__).resolve().parent
    old_data_path = base_dir / "0206data" / "train_val"
    new_data_path = base_dir / "Dataset001_lumber"

    data_module = DataModuleSafe(
        old_data_path=old_data_path,
        new_data_path=new_data_path,
        batch_size=2,
        seed=None,  # ランダム分割
    )

    model = MultiClassModel(
        in_channels=1,
        num_classes=3,
        encoder_name="efficientnet-b0",
        lr=1e-3,
        ce_weight=0.0,  # Dice lossのみ
    )

    dt = datetime.datetime.now()
    logger = TensorBoardLogger(
        "logs",
        name=dt.strftime("%Y-%m-%d_%H-%M-%S"),
        version="version_0",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="3d_v4",
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
    print("Training: Dice loss only, no CE, random split, 34 safe cases")
    trainer.fit(model, datamodule=data_module)
