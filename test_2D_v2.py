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

# GTに硬膜管(class 2)ラベルが1画素も存在しない症例。含めるとDiceが強制的に0になるため
# 参考値として除外した集計も別途出す（詳細はREADMEの「2D vs 3D 比較実験」注記を参照）
NO_DURAL_SAC_LABEL_CASES = {"00008"}

image_files = sorted(image_dir.glob("*.nii"))
print(f"テスト症例数: {len(image_files)}")

case_ids = []
case_dice_list = []

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

    # 3Dモデルの評価方法と統一するため、スライスごとではなく症例(ボリューム)全体を
    # 1サンプルとして混同行列を集計し、症例につき1つのDiceを算出する
    tp, fp, fn, tn = smp.metrics.get_stats(
        vol_preds.unsqueeze(0), vol_masks.unsqueeze(0), mode="multiclass", num_classes=3
    )  # [1, C]

    case_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")[0]  # [C]
    case_ids.append(base)
    case_dice_list.append(case_dice)
    print(f" → nerve={case_dice[1].item():.4f}, dural sac={case_dice[2].item():.4f}")

all_dice = torch.stack(case_dice_list)  # [N, C]

# 症例ごとのDiceを平均(3Dモデルと同じ「症例単位Diceの平均」で統一)
mean_dice = all_dice.mean(dim=0)

print()
print("=" * 45)
print("【2D U-Net テスト結果（症例単位Diceの平均、3Dモデルと同一の集計方法）】")
print(f"nerve(class 1)  Dice: {mean_dice[1].item():.4f}")
print(f"dural sac(class 2) Dice: {mean_dice[2].item():.4f}")
print(f"Overall Dice (nerve+dural sac 平均): {mean_dice[1:].mean().item():.4f}")
print("=" * 45)

# 参考: GTに硬膜管ラベルが無い症例を除いた場合
keep_mask = [cid not in NO_DURAL_SAC_LABEL_CASES for cid in case_ids]
if not all(keep_mask) and any(keep_mask):
    mean_dice_excl = all_dice[keep_mask].mean(dim=0)
    excluded = [cid for cid in case_ids if cid in NO_DURAL_SAC_LABEL_CASES]
    print(f"【参考】{excluded}を除いた{sum(keep_mask)}症例平均")
    print(f"nerve(class 1)  Dice: {mean_dice_excl[1].item():.4f}")
    print(f"dural sac(class 2) Dice: {mean_dice_excl[2].item():.4f}")
    print(f"Overall Dice (nerve+dural sac 平均): {mean_dice_excl[1:].mean().item():.4f}")
    print("=" * 45)
