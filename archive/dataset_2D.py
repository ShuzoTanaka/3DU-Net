import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class Nifti2DDataset(Dataset):
    def __init__(self, image_dir, mask_dir, slice_mode="middle", transform=None):
        """
        NIfTI形式（.nii, .nii.gz）の3D画像から2Dスライスを作るDataset

        Args:
            image_dir (str): 画像フォルダのパス
            mask_dir (str): マスクフォルダのパス
            slice_mode (str): "middle"なら中央スライス、"all"なら全スライス
            transform: 任意のtorchvision変換
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.slice_mode = slice_mode

        # 画像とマスクを対応付ける
        self.pairs = []
        for img_file in self.image_files:
            base = os.path.splitext(img_file)[0].replace(".nii", "")
            for mask_file in self.mask_files:
                if base in mask_file:
                    img_path = os.path.join(image_dir, img_file)
                    mask_path = os.path.join(mask_dir, mask_file)
                    img_nii = nib.load(img_path)
                    img_data = img_nii.get_fdata()
                    if slice_mode == "middle":
                        self.pairs.append((img_path, mask_path, img_data.shape[2] // 2))
                    elif slice_mode == "all":
                        for z in range(img_data.shape[2]):
                            self.pairs.append((img_path, mask_path, z))
                    break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, z = self.pairs[idx]

        # NIfTI読み込み
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        img = img_nii.get_fdata()
        mask = mask_nii.get_fdata()

        # 指定スライスを抽出
        img_2d = img[:, :, z]
        mask_2d = mask[:, :, z]

        # 正規化
        img_2d = img_2d.astype(np.float32)
        img_2d = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d) + 1e-8)

        img_2d = torch.tensor(img_2d).unsqueeze(0)  # [1, H, W]
        mask_2d = torch.tensor(mask_2d).long()  # [H, W]

        if self.transform:
            img_2d = self.transform(img_2d)

        return img_2d, mask_2d
