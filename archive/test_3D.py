import os
import torch
import nibabel as nib
import numpy as np
import segmentation_models_pytorch as smp
from lightning.pytorch import Trainer
from model import MultiClassModel
from dataModuleForTest import DataModuleForTest
import pandas as pd


def compute_dice_score(pred, target, threshold=0.5):
    """バイナリのDiceスコアを計算"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    if union == 0:
        return 1.0  # 両方ゼロの場合は完全一致とする
    return (2.0 * intersection) / union


def compute_dice_scores_3D(pred_path, target_path):
    """3D全体および各スライスごとのDiceを計算"""
    pred_nii = nib.load(pred_path)
    target_nii = nib.load(target_path)

    pred_vol = pred_nii.get_fdata()
    target_vol = target_nii.get_fdata()

    # 余分な次元を除去
    if pred_vol.ndim == 4 and pred_vol.shape[0] == 1:
        pred_vol = pred_vol[0]
    if target_vol.ndim == 4 and target_vol.shape[0] == 1:
        target_vol = target_vol[0]

    # 3D Dice
    dice_3d = compute_dice_score(torch.tensor(pred_vol), torch.tensor(target_vol))

    # 各スライスDice
    dice_2d_list = []
    for z in range(pred_vol.shape[0]):
        pred_slice = torch.tensor(pred_vol[z])
        target_slice = torch.tensor(target_vol[z])
        dice_2d_list.append(compute_dice_score(pred_slice, target_slice))

    mean_dice_2d = float(np.mean(dice_2d_list))
    return float(dice_3d), mean_dice_2d


if __name__ == "__main__":
    # モデル・データ設定
    model_path = r"C:\Users\orilab\Desktop\Tanaka\pytorchLightning\3d_1009\best-epoch=121-val_loss=0.15.ckpt"
    dataset_path = r"C:\Users\orilab\Desktop\Tanaka\pytorchLightning\temp0206"
    output_dir = "nifti_predictions0206_summary"

    os.makedirs(output_dir, exist_ok=True)

    model = MultiClassModel.load_from_checkpoint(model_path)
    data_module = DataModuleForTest(dataset_path=dataset_path, batch_size=1)
    data_module.setup()

    print(f"✅ Total test samples: {len(data_module.test_dataset)}")

    trainer = Trainer(accelerator="gpu", devices=1)
    trainer.test(model=model, datamodule=data_module)

    # === Dice計算 ===
    results = []
    for i in range(len(data_module.test_dataset)):
        for c in range(model.num_classes):
            pred_path = f"nifti_predictions0206_2/sample_{i}_class_{c}_pred.nii.gz"
            target_path = f"nifti_predictions0206_2/sample_{i}_class_{c}_gt.nii.gz"

            if not os.path.exists(pred_path) or not os.path.exists(target_path):
                continue

            dice_3d, dice_2d_mean = compute_dice_scores_3D(pred_path, target_path)
            results.append(
                {
                    "Sample": i,
                    "Class": c,
                    "Dice_3D": dice_3d,
                    "Mean_Dice_2D": dice_2d_mean,
                }
            )

            print(
                f"🧩 Sample {i} | Class {c} | 3D Dice: {dice_3d:.4f}, 2D mean Dice: {dice_2d_mean:.4f}"
            )

    # === CSV保存 ===
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "dice_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Diceスコア結果を保存しました → {csv_path}")
