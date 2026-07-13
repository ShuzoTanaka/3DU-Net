import torch
import numpy as np
import nibabel as nib
import segmentation_models_pytorch as smp
from pathlib import Path
from model_2D import MultiClassModel

BASE_DIR = Path(__file__).resolve().parent

# 3Dモデルと同じホールドアウト5症例でテスト（公平な比較）
model_path = BASE_DIR / "3d_2D_v2" / "best-2D-epoch=94-val_loss=0.08.ckpt"
test_data_path = BASE_DIR / "0206nii"

model = MultiClassModel.load_from_checkpoint(model_path)
model.eval()
model.cuda()

image_dir = test_data_path / "images"
mask_dir = test_data_path / "masks"

image_files = sorted(image_dir.glob("*.nii"))
print(f"テスト症例数: {len(image_files)}")

all_tp = None
all_fp = None
all_fn = None
all_tn = None

for img_path in image_files:
    base = img_path.stem
    mask_path = mask_dir / f"{base}.nii.gz"
    if not mask_path.exists():
        print(f"  マスクなし: {base}, スキップ")
        continue

    img_vol = nib.load(img_path).get_fdata().astype(np.float32)
    mask_vol = nib.load(mask_path).get_fdata().astype(np.int64)
    n_slices = img_vol.shape[2]

    print(f"  {base}: {n_slices} スライス", end="")

    vol_preds = []
    vol_masks = []

    with torch.no_grad():
        for z in range(n_slices):
            img_2d = img_vol[:, :, z]
            mask_2d = mask_vol[:, :, z]

            img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
            img_t = torch.tensor(img_2d).unsqueeze(0).unsqueeze(0).float().cuda()  # [1,1,H,W]

            pred = model(img_t)          # [1, 3, H, W]
            pred_label = pred.argmax(dim=1)  # [1, H, W]

            vol_preds.append(pred_label.cpu())
            vol_masks.append(torch.tensor(mask_2d).unsqueeze(0).long())

    vol_preds = torch.cat(vol_preds, dim=0)   # [D, H, W]
    vol_masks = torch.cat(vol_masks, dim=0)   # [D, H, W]

    tp, fp, fn, tn = smp.metrics.get_stats(
        vol_preds, vol_masks, mode="multiclass", num_classes=3
    )

    if all_tp is None:
        all_tp, all_fp, all_fn, all_tn = tp, fp, fn, tn
    else:
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_tn += tn

    case_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").mean(dim=0)
    print(f" → nerve={case_dice[1].item():.4f}, spinal={case_dice[2].item():.4f}")

# 全症例集計
mean_dice = smp.metrics.f1_score(all_tp, all_fp, all_fn, all_tn, reduction="none").mean(dim=0)

print()
print("=" * 45)
print("【2D U-Net テスト結果】")
print(f"nerve(class 1)  Dice: {mean_dice[1].item():.4f}")
print(f"spinal(class 2) Dice: {mean_dice[2].item():.4f}")
print(f"Overall Dice (nerve+spinal 平均): {mean_dice[1:].mean().item():.4f}")
print("=" * 45)
