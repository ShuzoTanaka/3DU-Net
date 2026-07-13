# archive/

現在のパイプライン（ルート直下の `train_v4.py` / `train_2D_v2.py` / `test.py` / `test_2D_v2.py` など）に統合される前の、
初期・試作段階のスクリプトを保管している。動作の参考や経緯の確認以外で使う必要はない。

| ファイル | 内容 | 現行の代替 |
|---|---|---|
| `train.py` | 最初期の3D学習スクリプト | `train_v4.py` |
| `train_2D.py` | 最初期の2D学習スクリプト | `train_2D_v2.py` |
| `test_2D.py`, `test_2D_3D_dice.py`, `test_3D.py` | 初期のテストスクリプト（ホールドアウト管理なし） | `test.py`, `test_2D_v2.py` |
| `dataModule_2D.py`, `dataset_2D.py` | 初期の2Dデータモジュール（データリーク対策なし） | `DataModule2DSafe.py` |
| `dataModuleForTrain.py` | 未使用の初期データモジュール | `DataModuleSafe.py` |
| `DataModuleCombined.py` | データリークに気づく前の単純結合版データモジュール（参考用：[README](../README.md)のデータリーク修正の経緯を示す） | `DataModuleSafe.py` |
| `dice_coefficient.py` | 自前のDice計算実装（後に`segmentation_models_pytorch`の`smp.metrics`に置き換え） | `model.py`内の`smp.metrics` |
| `inference_2D.py`, `save_predictions.py` | 初期の推論・予測保存スクリプト | `test_2D_v2.py`, `model.py`の`test_step` |
