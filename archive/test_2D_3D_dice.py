import os
import torch
import nibabel as nib
import numpy as np
from lightning.pytorch import Trainer
from model_2D import MultiClassModel
from dataset_2D import Nifti2DDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from PIL import Image


def dice_coefficient(pred, target, eps=1e-6):
    """2D Dice係数を手動で計算"""
    pred = pred > 0.5
    target = target > 0.5
    intersection = (pred & target).sum().float()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def calculate_dice_2d_and_3d(model, dataset_path, target_class=1, output_dir="results"):
    """
    2DスライスDiceと3DボリュームDiceを両方計算
    """
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(dataset_path, "images")
    mask_dir = os.path.join(dataset_path, "masks")

    model.eval().cuda()

    dice_2d_all = []
    dice_3d_all = []

    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith(".nii"):
            continue

        base = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)

        # 対応するマスク探す (.nii or .nii.gz)
        mask_path = None
        for ext in [".nii", ".nii.gz"]:
            candidate = os.path.join(mask_dir, base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            print(f"[WARN] Mask not found for {fname}")
            continue

        # NIfTI読み込み
        img_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        img = img_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.int64)

        # 正規化
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

        pred_volume = np.zeros_like(mask, dtype=np.float32)
        dice_per_slice = []

        for z in range(img.shape[2]):
            img_2d = (
                torch.tensor(img[:, :, z]).unsqueeze(0).unsqueeze(0).cuda()
            )  # [1, 1, H, W]

            with torch.no_grad():
                pred_2d = model(img_2d)
                pred_2d = torch.sigmoid(pred_2d)[0, target_class, :, :].cpu().numpy()

            true_2d = (mask[:, :, z] == target_class).astype(np.float32)
            pred_volume[:, :, z] = pred_2d

            # 2D Dice
            d2 = dice_coefficient(torch.tensor(pred_2d), torch.tensor(true_2d))
            dice_per_slice.append(d2.item())

        # 3D Dice（スライス全体を統合して1つのボリュームで計算）
        pred_3d = pred_volume > 0.5
        true_3d = mask == target_class
        intersection = np.logical_and(pred_3d, true_3d).sum()
        dice_3d = (2 * intersection + 1e-6) / (pred_3d.sum() + true_3d.sum() + 1e-6)

        # 平均Dice（2D）
        mean_2d = np.mean(dice_per_slice)
        dice_2d_all.append(mean_2d)
        dice_3d_all.append(dice_3d)

        print(f"--- {fname} ---")
        print(f"Mean 2D Dice: {mean_2d:.4f}")
        print(f"3D Volume Dice: {dice_3d:.4f}")

        # 結果の1枚目を保存
        mid_z = img.shape[2] // 2
        Image.fromarray((img[:, :, mid_z] * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f"{base}_image.png")
        )
        Image.fromarray((pred_volume[:, :, mid_z] > 0.5).astype(np.uint8) * 255).save(
            os.path.join(output_dir, f"{base}_pred.png")
        )
        Image.fromarray(
            ((mask[:, :, mid_z] == target_class).astype(np.uint8)) * 255
        ).save(os.path.join(output_dir, f"{base}_true.png"))

    print("\n==== Overall Results ====")
    print(f"Average 2D Dice (all slices): {np.mean(dice_2d_all):.4f}")
    print(f"Average 3D Dice (per volume): {np.mean(dice_3d_all):.4f}")


if __name__ == "__main__":
    # 学習済みモデル読み込み
    model_path = r"C:\Users\orilab\Desktop\Tanaka\pytorchLightning\3d_1009\best-epoch=121-val_loss=0.15.ckpt"
    model = MultiClassModel.load_from_checkpoint(model_path)
    model.cuda().eval()

    dataset_path = "temp0206"  # NIfTIデータフォルダ
    calculate_dice_2d_and_3d(
        model, dataset_path, target_class=1, output_dir="results0206"
    )
