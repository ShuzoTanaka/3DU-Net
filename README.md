# 3D Medical Image Segmentation — Nerve & Spinal Cord

3D U-Net による腰椎 MRI の神経・脊髄自動セグメンテーション。PyTorch Lightning で実装。

## 結果 (v5 最終モデル)

| クラス | Dice係数 |
|---|---|
| 背景 (class 0) | 0.9993 |
| **神経 nerve (class 1)** | **0.6794** |
| **脊髄 spinal (class 2)** | **0.6768** |
| **nerve + spinal 平均** | **0.6781** |

テストデータ: 5症例（学習に未使用のホールドアウト症例）  
学習データ: 34症例（安全版・データリーク無し）

## モデル構成

| 項目 | 設定 |
|---|---|
| アーキテクチャ | 3D U-Net |
| エンコーダ | EfficientNet-B0 |
| 入力チャネル | 1 (グレースケール) |
| 出力クラス | 3 (背景 / 神経 / 脊髄) |
| Loss 関数 | Dice Loss (multiclass) |
| オプティマイザ | Adam (lr=1e-3) |
| スケジューラ | CosineAnnealingLR (T_max=200) |
| エポック数 | 200 |
| バッチサイズ | 2 |
| 入力サイズ | 256×256×64 |
| フレームワーク | PyTorch Lightning 2.4.0 |

## ファイル構成

```
pytorchLightning/
├── model.py              # MultiClassModel (3D U-Net, Dice/CE loss対応)
├── dataset.py            # NiftiDataset, NnUNetDataset (データ読み込み・拡張)
├── DataModuleSafe.py     # 安全版データモジュール (ホールドアウト除外)
├── DataModuleCombined.py # 参照用・リーク版データモジュール
├── dataModule.py         # 既存データのみのデータモジュール
├── dataModuleForTest.py  # テスト専用データモジュール
├── train_v5.py           # 最終学習スクリプト (Dice only, 34症例, random split)
├── train_v4.py           # 安全版 (Dice+CE, seed=42)
├── train_v3.py           # 参照用 (データリークあり・不正)
├── test.py               # テストスクリプト (v5チェックポイント使用)
└── .gitignore
```

## セットアップ

```bash
python -m venv .lightningenv
.lightningenv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install lightning segmentation_models_pytorch_3d nibabel
```

## 学習

```bash
cd pytorchLightning
python train_v5.py
```

## テスト

```bash
python test.py
```

## データ仕様

- 入力画像: NIfTI形式 (`.nii` または `.nii.gz`)
- ラベル: 0=背景, 1=神経, 2=脊髄
- nnUNet形式にも対応 (`*_0000.nii.gz` / `*.nii.gz`)

## バージョン比較

| バージョン | 設定 | nerve | spinal | 平均 | 信頼性 |
|---|---|---|---|---|---|
| v4 | CE loss + seed=42, 34症例 | 0.645 | 0.544 | 0.595 | ◎ |
| **v5** | **Dice only + random split, 34症例** | **0.679** | **0.677** | **0.678** | **◎** |
| v3 (参照) | データリークあり (不正) | 0.734 | 0.868 | 0.801 | × |

## 環境

- GPU: NVIDIA GeForce RTX 4080
- CUDA: 12.1
- PyTorch: 2.5.1+cu121
- Python: 3.11.6
