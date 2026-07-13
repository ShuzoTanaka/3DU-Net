from pathlib import Path
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset_2D import Nifti2DDataset
import lightning as L


class DataModule(L.LightningDataModule):
    def __init__(self, dataset_path, batch_size=2):
        """
        PyTorch Lightning DataModule for loading PNG datasets.

        Args:
            dataset_path (str): Path to the dataset directory.
            batch_size (int): Number of samples per batch.
        """
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size

    def setup(self):
        self.test_dataset = Nifti2DDataset(
            image_dir=os.path.join(self.dataset_path, "images"),
            mask_dir=os.path.join(self.dataset_path, "masks"),
            slice_mode="middle",
        )

    def train_dataloader(self):
        """
        Return DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        """
        Return DataLoader for validation data.
        """
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )

    def test_dataloader(self):
        """
        Return DataLoader for testing data.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )


if __name__ == "__main__":
    # データモジュールの動作確認
    dataset_path = r"C:\Users\orilab\Desktop\Tanaka\pytorchLightning\data2"
    batch_size = 4

    dm = DataModule(dataset_path, batch_size)
    dm.setup()

    print("Train DataLoader:")
    for images, masks in dm.train_dataloader():
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        break  # 1バッチだけ確認

    print("Validation DataLoader:")
    for images, masks in dm.val_dataloader():
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        break  # 1バッチだけ確認

    print("Test DataLoader:")
    for images, masks in dm.test_dataloader():
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        break  # 1バッチだけ確認
