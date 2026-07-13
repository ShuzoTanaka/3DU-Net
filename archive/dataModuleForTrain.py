# 2/6作成　トレーニング用データモジュール
# テストデータと完全に分けて学習を行いたいとき用、データセットをtrainとvalにのみ分割する

from pathlib import Path
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
        dataset = NiftiDataset(image_folder, mask_folder)
        total_size = len(dataset)
        # Split into train, validation, and test sets
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        # # 1症例test用
        # self.test_dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False
        )
