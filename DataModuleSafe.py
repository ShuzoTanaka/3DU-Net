from pathlib import Path
import torch
import lightning as L
from torch.utils.data import DataLoader, random_split, ConcatDataset
from dataset import NiftiDataset, NnUNetDataset

# ホールドアウトテスト症例 (0206nii/images に対応するケース番号)
# これらはDataset001_lumberに含まれているが学習に使ってはいけない
HOLDOUT_CASES = {"case003", "case006", "case008", "case010", "case011"}


class NnUNetDatasetSafe(NnUNetDataset):
    """ホールドアウト症例を除外したNnUNetDataset"""

    def __init__(self, image_folder, label_folder, exclude_cases=None, target_shape=(256, 256, 64), augment=False):
        self.target_shape = target_shape
        self.augment = augment
        exclude = exclude_cases or set()

        image_paths_raw = sorted(Path(image_folder).glob("*_0000.nii.gz"))
        self.image_paths = []
        self.label_paths = []
        for img_path in image_paths_raw:
            case_name = img_path.name.replace("_0000.nii.gz", "")
            if case_name in exclude:
                print(f"  [除外] {case_name} (ホールドアウトテスト症例)")
                continue
            lbl_path = Path(label_folder) / f"{case_name}.nii.gz"
            if lbl_path.exists():
                self.image_paths.append(img_path)
                self.label_paths.append(lbl_path)


class DataModuleSafe(L.LightningDataModule):
    """
    安全な統合データモジュール：ホールドアウト症例を完全除外。
    - 既存データ (0206data/train_val): 25症例
    - Dataset001_lumber の新規症例のみ: case032-040 から holdout除外後 → 9症例
    - 合計: 34症例（テスト漏れなし）
    """

    def __init__(self, old_data_path, new_data_path, batch_size=2, seed=None):
        super().__init__()
        self.old_data_path = Path(old_data_path)
        self.new_data_path = Path(new_data_path)
        self.batch_size = batch_size
        self.seed = seed  # None=ランダム分割（元モデルと同じ）、整数=再現性あり

    def setup(self, stage=None):
        print("Preparing safe combined dataset (holdout cases excluded)...")

        # --- 既存データ (0206data/train_val) ---
        old_img = self.old_data_path / "images"
        old_mask = self.old_data_path / "masks"
        old_train = NiftiDataset(old_img, old_mask, augment=True)
        old_val = NiftiDataset(old_img, old_mask, augment=False)

        # --- Dataset001_lumber: 既存25症例(case001-031)は0206dataと重複するため除外 ---
        # 新規のみ使用: case032, 033, 034, 035, 036, 037, 038, 039, 040
        # ただしテストケース(case003,006,008,010,011)は除外
        # imagesTr から新規症例 (case032, 033, 034, 035, 038)
        # imagesTs から新規症例 (case036, 037, 039, 040) ← case003,006,010,011はホールドアウト
        existing_cases = {f"case{i:03d}" for i in [
            1, 2, 4, 5, 7, 9, 12, 13, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ]}
        exclude_tr = existing_cases | HOLDOUT_CASES  # 既存25 + holdout8 を除外

        new_tr_img = self.new_data_path / "imagesTr"
        new_tr_lbl = self.new_data_path / "labelsTr"
        new_ts_img = self.new_data_path / "imagesTs"
        new_ts_lbl = self.new_data_path / "labelsTs"

        new_tr_aug = NnUNetDatasetSafe(new_tr_img, new_tr_lbl, exclude_cases=exclude_tr, augment=True)
        new_tr_noaug = NnUNetDatasetSafe(new_tr_img, new_tr_lbl, exclude_cases=exclude_tr, augment=False)
        new_ts_aug = NnUNetDatasetSafe(new_ts_img, new_ts_lbl, exclude_cases=HOLDOUT_CASES, augment=True)
        new_ts_noaug = NnUNetDatasetSafe(new_ts_img, new_ts_lbl, exclude_cases=HOLDOUT_CASES, augment=False)

        print(f"  既存データ: {len(old_train)} 症例")
        print(f"  Dataset001_lumber 新規 (imagesTr): {len(new_tr_aug)} 症例")
        print(f"  Dataset001_lumber 新規 (imagesTs): {len(new_ts_aug)} 症例")
        total = len(old_train) + len(new_tr_aug) + len(new_ts_aug)
        print(f"  合計 (ホールドアウト除外済み): {total} 症例")

        combined_aug = ConcatDataset([old_train, new_tr_aug, new_ts_aug])
        combined_noaug = ConcatDataset([old_val, new_tr_noaug, new_ts_noaug])

        total_size = len(combined_aug)
        train_size = int(0.85 * total_size)
        val_size = total_size - train_size

        if self.seed is not None:
            gen = torch.Generator().manual_seed(self.seed)
            gen2 = torch.Generator().manual_seed(self.seed)
        else:
            gen = None
            gen2 = None

        self.train_dataset, _ = random_split(combined_aug, [train_size, val_size], generator=gen)
        _, self.val_dataset = random_split(combined_noaug, [train_size, val_size], generator=gen2)

        print(f"  Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
