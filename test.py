# 3Dnifti用testコード　使える！！

import torch
from lightning.pytorch import Trainer
from model import MultiClassModel  # モデル定義
from dataModuleForTest import DataModuleForTest  # テスト用のデータモジュール

# モデルとデータモジュールのインスタンス化
model_path = r"C:\Users\orilab\Desktop\Tanaka\pytorchLightning\3d_v5\best-epoch=182-val_loss=0.13.ckpt"  # v5: Dice only, 34症例, no seed（正式版）
model = MultiClassModel.load_from_checkpoint(model_path)

# データモジュールのインスタンス化
data_module = DataModuleForTest(dataset_path="0206nii", batch_size=2)

# ここでセットアップを実行
data_module.setup()

# テストデータセットの数を確認
print(f"Total test samples: {len(data_module.test_dataset)}")

# テストエポックの実行
trainer = Trainer(accelerator="gpu", devices=1)
trainer.test(model=model, datamodule=data_module)
