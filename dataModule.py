from pathlib import Path
import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
from dataset import NiftiDataset


class DataModule(L.LightningDataModule):
    def __init__(self, dataset_path, batch_size=2):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        # self.dataset_path = dataset_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        print("Preparing data...")
        image_folder = self.dataset_path / "images"
        mask_folder = self.dataset_path / "masks"
        # val/test用はaugment=False、train用はaugment=True
        dataset_no_aug = NiftiDataset(image_folder, mask_folder, augment=False)
        dataset_aug = NiftiDataset(image_folder, mask_folder, augment=True)
        total_size = len(dataset_no_aug)
        # Split into train, validation, and test sets
        train_val_size = int(0.8 * total_size)
        test_size = total_size - train_val_size
        train_size = int(0.8 * train_val_size)
        val_size = train_val_size - train_size

        # 再現性のためseedを固定
        generator = torch.Generator().manual_seed(42)
        generator_aug = torch.Generator().manual_seed(42)

        train_val_dataset_no_aug, self.test_dataset = random_split(
            dataset_no_aug, [train_val_size, test_size], generator=generator
        )
        train_val_dataset_aug, _ = random_split(
            dataset_aug, [train_val_size, test_size], generator=generator_aug
        )

        _, self.val_dataset = random_split(
            train_val_dataset_no_aug, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        self.train_dataset, _ = random_split(
            train_val_dataset_aug, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # # 1症例test用
        # self.test_dataset = dataset_no_aug

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

    def test_dataloader(self):
        # テストデータローダーの作成
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False
        )
