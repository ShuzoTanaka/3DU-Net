from pathlib import Path
import torch
import numpy as np
import nibabel as nib
import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

HOLDOUT_CASES = {"case003", "case006", "case008", "case010", "case011"}


class Nifti2DSliceDataset(Dataset):
    """NIfTIボリュームから全2Dスライスを生成するDataset"""

    def __init__(self, slice_list, augment=False):
        self.slice_list = slice_list  # [(img_path, mask_path, z), ...]
        self.augment = augment

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, idx):
        img_path, mask_path, z = self.slice_list[idx]
        img = nib.load(img_path).get_fdata()[:, :, z].astype(np.float32)
        mask = nib.load(mask_path).get_fdata()[:, :, z].astype(np.int64)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_t = torch.tensor(img).unsqueeze(0)   # [1, H, W]
        mask_t = torch.tensor(mask).long()        # [H, W]

        if self.augment:
            if torch.rand(1) > 0.5:
                img_t = torch.flip(img_t, [1])
                mask_t = torch.flip(mask_t, [0])
            if torch.rand(1) > 0.5:
                img_t = torch.flip(img_t, [2])
                mask_t = torch.flip(mask_t, [1])

        return img_t, mask_t


def _build_slice_list_nifti(image_dir, mask_dir):
    """0206data形式 (*.nii → *.nii.gz) のスライスリストを生成"""
    slices = []
    for img_path in sorted(Path(image_dir).glob("*.nii")):
        base = img_path.stem  # e.g. "00001"
        mask_path = Path(mask_dir) / f"{base}.nii.gz"
        if not mask_path.exists():
            continue
        n_slices = nib.load(img_path).shape[2]
        for z in range(n_slices):
            slices.append((str(img_path), str(mask_path), z))
    return slices


def _build_slice_list_nnunet(image_dir, label_dir, exclude_cases=None):
    """nnUNet形式 (*_0000.nii.gz → *.nii.gz) のスライスリストを生成"""
    exclude = exclude_cases or set()
    slices = []
    for img_path in sorted(Path(image_dir).glob("*_0000.nii.gz")):
        case_name = img_path.name.replace("_0000.nii.gz", "")
        if case_name in exclude:
            print(f"  [除外] {case_name}")
            continue
        mask_path = Path(label_dir) / f"{case_name}.nii.gz"
        if not mask_path.exists():
            continue
        n_slices = nib.load(img_path).shape[2]
        for z in range(n_slices):
            slices.append((str(img_path), str(mask_path), z))
    return slices


class DataModule2DSafe(L.LightningDataModule):
    """
    2D安全版データモジュール：全スライス使用・ホールドアウト除外
    - 0206data/train_val: 25症例
    - Dataset001_lumber 新規のみ: 9症例
    - 合計: 34症例 → 全スライスで約1900枚以上
    """

    def __init__(self, old_data_path, new_data_path, batch_size=16, seed=None):
        super().__init__()
        self.old_data_path = Path(old_data_path)
        self.new_data_path = Path(new_data_path)
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage=None):
        print("Preparing 2D safe dataset (all slices, holdout excluded)...")

        # 0206data: 全スライス
        old_slices = _build_slice_list_nifti(
            self.old_data_path / "images",
            self.old_data_path / "masks",
        )

        # Dataset001_lumber: 既存25症例 + holdout5症例を除外
        existing_cases = {f"case{i:03d}" for i in [
            1, 2, 4, 5, 7, 9, 12, 13, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ]}
        exclude_tr = existing_cases | HOLDOUT_CASES
        new_tr_slices = _build_slice_list_nnunet(
            self.new_data_path / "imagesTr",
            self.new_data_path / "labelsTr",
            exclude_cases=exclude_tr,
        )
        new_ts_slices = _build_slice_list_nnunet(
            self.new_data_path / "imagesTs",
            self.new_data_path / "labelsTs",
            exclude_cases=HOLDOUT_CASES,
        )

        all_slices = old_slices + new_tr_slices + new_ts_slices
        print(f"  0206data:              {len(old_slices)} スライス")
        print(f"  Dataset001_lumber新規(Tr): {len(new_tr_slices)} スライス")
        print(f"  Dataset001_lumber新規(Ts): {len(new_ts_slices)} スライス")
        print(f"  合計:                  {len(all_slices)} スライス")

        total = len(all_slices)
        train_size = int(0.85 * total)
        val_size = total - train_size

        gen = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        gen2 = torch.Generator().manual_seed(self.seed) if self.seed is not None else None

        aug_dataset = Nifti2DSliceDataset(all_slices, augment=True)
        noaug_dataset = Nifti2DSliceDataset(all_slices, augment=False)

        self.train_dataset, _ = random_split(aug_dataset, [train_size, val_size], generator=gen)
        _, self.val_dataset = random_split(noaug_dataset, [train_size, val_size], generator=gen2)

        print(f"  Train: {len(self.train_dataset)} スライス, Val: {len(self.val_dataset)} スライス")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
