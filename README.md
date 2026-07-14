# Medical Image Segmentation — Nerve & Dural Sac in Lumbar MRI

腰椎MRIから神経根（nerve）・硬膜管（dural sac）を自動セグメンテーションする多クラスセグメンテーションモデル。PyTorch Lightningで実装し、3D U-Netと2D U-Netの両方を学習・比較した上で、下流のTractographyパイプラインの要件を踏まえて**2D U-Netを採用モデル**として選定した。

## 採用モデル: 2D U-Net

| クラス | Dice係数 |
|---|---|
| **神経根 nerve (class 1)** | **0.7399** |
| 硬膜管 dural sac (class 2) | 0.5548 |
| nerve + dural sac 平均 | 0.6473 |

採用理由は次の「2D vs 3D 比較実験」を参照。

## 2D vs 3D 比較実験

同一の34症例・同一のホールドアウト5症例でテストした結果。**Diceは両モデルとも「症例(ボリューム)全体を1つの混同行列として集計し、症例ごとに算出したDiceを5症例で均等平均する」方法に統一している**。これは2段階の修正を経て統一したもの：

1. 2Dは元々スライスごとにDiceを算出して平均しており、3Dの症例単位集計と方法が異なっていたため、`test_2D_v2.py`を修正して症例単位に揃えた
2. さらに3D側（`model.py`）にも、バッチ単位でDiceを平均してからバッチ間で平均するというバグがあった。ホールドアウトは5症例・バッチサイズ2のため`[2症例, 2症例, 1症例]`に分割され、最後の1症例だけのバッチが他のバッチと同じ重みでカウントされ、症例間の重みが不均等になっていた。`model.py`を修正し、バッチサイズによらず全症例を均等平均するようにした

| モデル | nerve Dice | dural sac Dice | 平均 |
|---|---|---|---|
| **2D U-Net（採用）** | **0.7399** | 0.5548 | 0.6473 |
| 3D U-Net | 0.6798 | **0.6518** | **0.6658** |

**参考：ホールドアウト症例のうち1例（00008）はGTラベルに硬膜管クラスの画素が1つも存在せず、この症例の硬膜管Diceは強制的に0になる。この症例を除いた4症例の平均：**

| モデル | nerve Dice | dural sac Dice | 平均 |
|---|---|---|---|
| 2D U-Net | 0.7257 | 0.6934 | 0.7095 |
| 3D U-Net | 0.6773 | 0.8148 | 0.7460 |

**考察：**
- 神経根（nerve）は2Dモデルが優位 → 各スライスに点状に現れる局所構造の検出が得意
- 硬膜管（dural sac）は3Dモデルが優位 → 複数スライスにまたがる連続構造の把握に3Dが有効（00008を除くとさらに差が開く）
- 平均Diceだけを見れば3Dモデルの方が高いが、後段のTractographyパイプラインでは神経根ROIの検出が解析の起点となるため、**平均値ではなく神経根の精度を優先し2Dモデルの採用が合理的と判断した**

## モデル構成

| 項目 | 2D U-Net（採用モデル） | 3D U-Net（比較検証用） |
|---|---|---|
| アーキテクチャ | 2D U-Net | 3D U-Net |
| エンコーダ | EfficientNet-B0 | EfficientNet-B0 |
| 入力チャネル | 1 (グレースケール) | 1 (グレースケール) |
| 出力クラス | 3 (背景 / 神経根 / 硬膜管) | 3 (背景 / 神経根 / 硬膜管) |
| Loss 関数 | Dice Loss (multiclass) | Dice Loss (multiclass) |
| オプティマイザ | Adam (lr=1e-3) | Adam (lr=1e-3) + CosineAnnealingLR (T_max=200) |
| エポック数 | 100 | 200 |
| バッチサイズ | 16 | 2 |
| 入力サイズ | スライス単位 (2D) | 256×256×64 |
| 学習スクリプト | `train_2D_v2.py` | `train_v4.py` |
| テストスクリプト | `test_2D_v2.py` | `test.py` |
| フレームワーク | PyTorch Lightning 2.4.0 | PyTorch Lightning 2.4.0 |

学習データ・テストデータは共通：学習34症例（安全版・データリーク無し）、テスト5症例（学習に未使用のホールドアウト症例）。

## ファイル構成

このリポジトリのルート直下がそのままプロジェクトルート（サブフォルダに入っていない）。

```
.
├── model.py              # MultiClassModel (3D U-Net, Dice/CE loss対応)
├── model_2D.py           # MultiClassModel (2D U-Net)
├── dataset.py            # NiftiDataset, NnUNetDataset (データ読み込み・拡張)
├── DataModuleSafe.py     # 3D安全版データモジュール (ホールドアウト除外)
├── DataModule2DSafe.py   # 2D版データモジュール (全スライス, ホールドアウト除外)
├── dataModule.py         # 既存データのみのデータモジュール
├── dataModuleForTest.py  # テスト専用データモジュール
├── train_v4.py           # 3D学習スクリプト (Dice only, 34症例, random split)
├── train_v3.py           # 3D比較用 (Dice+CE, seed=42)
├── train_v2.py           # Fine-tuning試験版
├── train_2D_v2.py        # 2D U-Net学習スクリプト (全スライス, 34症例、採用モデル)
├── test.py               # 3Dテストスクリプト
├── test_2D_v2.py         # 2Dテストスクリプト (ホールドアウト5症例)
├── nifti_to_png.py       # NIfTI→PNG変換ユーティリティ（可視化・発表資料作成用）
├── archive/              # 初期・試作段階のスクリプト（詳細はarchive/README.md参照）
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
# 採用モデル（2D U-Net）
python train_2D_v2.py

# 比較検証用（3D U-Net）
python train_v4.py
```

## テスト

```bash
# 採用モデル（2D U-Net）
python test_2D_v2.py

# 比較検証用（3D U-Net）
python test.py
```

## データ仕様

- 入力画像: NIfTI形式 (`.nii` または `.nii.gz`)
- ラベル: 0=背景, 1=神経根, 2=硬膜管
- nnUNet形式にも対応 (`*_0000.nii.gz` / `*.nii.gz`)

## 3Dモデルの学習設定比較（参考）

3D U-Net単体での設定違いによる精度比較（採用モデルの選定には直接関係しないが、Loss構成・分割方法の検証記録として残す）：

| バージョン | 設定 | nerve | dural sac | 平均 |
|---|---|---|---|---|
| v3 | Dice+CE loss, seed=42, 34症例 | 0.645 | 0.544 | 0.595 |
| v4 | Dice only, random split, 34症例 | 0.679 | 0.677 | 0.678 |

※ このv3/v4比較は「2D vs 3D 比較実験」で修正した集計バグ（バッチ単位平均によるDice算出）を修正する前の数値であり、再検証していない。傾向（Dice単体+ランダム分割が優位）は変わらないと考えられるが、正確な値としては参考程度に留めること。

## 環境

- GPU: NVIDIA GeForce RTX 4080
- CUDA: 12.1
- PyTorch: 2.5.1+cu121
- Python: 3.11.6
