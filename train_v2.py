import torch
from pathlib import Path
import datetime
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from dataModule import DataModule
from model import MultiClassModel

# 元の良いモデル(val_loss=0.15)からAugmentation付きでファインチューニング
if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    dataset_path = Path(
        "C:/Users/orilab/Desktop/Tanaka/pytorchLightning/0206data/train_val"
    )
    data_module = DataModule(dataset_path=dataset_path, batch_size=2)

    # 元の良いチェックポイントからウェイトをロード (lr=1e-4でファインチューニング)
    old_ckpt = r"C:\Users\orilab\Desktop\Tanaka\pytorchLightning\3d_1009\best-epoch=121-val_loss=0.15.ckpt"
    model = MultiClassModel.load_from_checkpoint(
        old_ckpt,
        in_channels=1,
        num_classes=3,
        encoder_name="efficientnet-b0",
        lr=1e-4,  # ファインチューニング用に低めのLR
    )

    dt = datetime.datetime.now()
    logger = TensorBoardLogger(
        "logs",
        name=dt.strftime("%Y-%m-%d_%H-%M-%S"),
        version="version_0",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="3d_finetune",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=150,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
    )

    data_module.setup()
    print("Fine-tuning from old checkpoint with Augmentation + CE loss + LR=1e-4")
    trainer.fit(model, datamodule=data_module)
