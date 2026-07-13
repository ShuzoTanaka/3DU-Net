# 3D U-Net テストスクリプト（train_v4.pyで学習した最終モデル用、README記載のDice 0.678はこの結果）

from pathlib import Path

from lightning.pytorch import Trainer
from model import MultiClassModel  # モデル定義
from dataModuleForTest import DataModuleForTest  # テスト用のデータモジュール

BASE_DIR = Path(__file__).resolve().parent

# 学習時のスクリプト名はtrain_v5.pyだったため（後にtrain_v4.pyへ改名）、
# チェックポイントの保存先フォルダ名は3d_v5のまま残っている
model_path = BASE_DIR / "3d_v5" / "best-epoch=182-val_loss=0.13.ckpt"
model = MultiClassModel.load_from_checkpoint(model_path)

# データモジュールのインスタンス化
data_module = DataModuleForTest(dataset_path=BASE_DIR / "0206nii", batch_size=2)

# ここでセットアップを実行
data_module.setup()

# テストデータセットの数を確認
print(f"Total test samples: {len(data_module.test_dataset)}")

# テストエポックの実行
trainer = Trainer(accelerator="gpu", devices=1)
trainer.test(model=model, datamodule=data_module)
