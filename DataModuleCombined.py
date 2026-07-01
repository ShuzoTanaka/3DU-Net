from pathlib import Path
import torch
import lightning as L
from torch.utils.data import DataLoader, random_split, ConcatDataset
from dataset import NiftiDataset, NnUNetDataset


class DataModuleCombined(L.LightningDataModule):
    """
    既存データ (0206data/train_val) + Dataset001_lumber を合わせた学習用DataModule。
    テストはDataModuleForTestで別途行う（0206nii/ ホールドアウト固定）。
    """

    def __init__(self, old_data_path, new_data_path, batch_size=2):
        super().__init__()
        self.old_data_path = Path(old_data_path)
        self.new_data_path = Path(new_data_path)
        self.batch_size = batch_size

    def setup(self, stage=None):
        print("Preparing combined dataset...")

        # --- 既存データ (0206data/train_val) ---
        old_img = self.old_data_path / "images"
        old_mask = self.old_data_path / "masks"
        old_train = NiftiDataset(old_img, old_mask, augment=True)
        old_val = NiftiDataset(old_img, old_mask, augment=False)

        # --- Dataset001_lumber (imagesTr + imagesTs) ---
        new_train_img = self.new_data_path / "imagesTr"
        new_train_lbl = self.new_data_path / "labelsTr"
        new_ts_img = self.new_data_path / "imagesTs"
        new_ts_lbl = self.new_data_path / "labelsTs"

        new_train_aug = NnUNetDataset(new_train_img, new_train_lbl, augment=True)
        new_train_noaug = NnUNetDataset(new_train_img, new_train_lbl, augment=False)
        new_ts_aug = NnUNetDataset(new_ts_img, new_ts_lbl, augment=True)
        new_ts_noaug = NnUNetDataset(new_ts_img, new_ts_lbl, augment=False)

        print(f"  既存データ: {len(old_train)} 症例")
        print(f"  Dataset001 Train: {len(new_train_aug)} 症例")
        print(f"  Dataset001 Test:  {len(new_ts_aug)} 症例")
        total = len(old_train) + len(new_train_aug) + len(new_ts_aug)
        print(f"  合計: {total} 症例")

        # 全データを合わせてtrain/valに分割 (seed固定)
        combined_aug = ConcatDataset([old_train, new_train_aug, new_ts_aug])
        combined_noaug = ConcatDataset([old_val, new_train_noaug, new_ts_noaug])

        total_size = len(combined_aug)
        train_size = int(0.85 * total_size)
        val_size = total_size - train_size

        gen = torch.Generator().manual_seed(42)
        self.train_dataset, _ = random_split(combined_aug, [train_size, val_size], generator=gen)

        gen2 = torch.Generator().manual_seed(42)
        _, self.val_dataset = random_split(combined_noaug, [train_size, val_size], generator=gen2)

        print(f"  Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
