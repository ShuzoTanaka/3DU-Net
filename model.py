import lightning as L
import torch
import segmentation_models_pytorch_3d as smp
import os
import nibabel as nib
import numpy as np
import torch.nn.functional as F


class MultiClassModel(L.LightningModule):
    def __init__(self, in_channels=1, num_classes=3, encoder_name="efficientnet-b0", lr=1e-3, ce_weight=0.5):
        super().__init__()
        self.lr = lr
        self.ce_weight = ce_weight  # 0.0でCE lossを無効化
        self.model = smp.Unet(
            encoder_name=encoder_name,  # エンコーダ（例: efficientnet-b0）
            in_channels=in_channels,  # 入力チャネル数（グレースケール=1）
            classes=num_classes,  # 出力クラス数
        )
        self.loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce_loss_fn = torch.nn.CrossEntropyLoss()
        self.test_outputs = []  # test_outputs を初期化

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=self.lr * 0.01
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def training_step(self, batch, batch_idx):
        images, masks = batch  # masksはクラスインデックス（0, 1, 2）で構成
        predictions = self.model(images)  # 出力形状: [B, 3, H, W, D]
        dice_loss = self.loss_fn(predictions, masks)
        ce_loss = self.ce_loss_fn(predictions, masks.squeeze(1).long())
        loss = dice_loss + self.ce_weight * ce_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        predictions = self.model(images)
        loss = self.loss_fn(predictions, masks)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        predictions = self(images)

        # オリジナルサイズにリサイズ
        original_shape = images.shape[2:]
        predictions = F.interpolate(predictions, size=original_shape, mode="nearest")

        # Dice計算用: クラスインデックス [B, H, W, D]
        masks_labels = masks.squeeze(1).long()
        pred_labels = predictions.argmax(dim=1)  # argmaxで予測クラスを決定

        # NIfTI保存用: one-hot [B, C, H, W, D]
        masks_onehot = F.one_hot(masks_labels, num_classes=predictions.shape[1])
        masks_onehot = masks_onehot.permute(0, 4, 1, 2, 3)
        pred_onehot = F.one_hot(pred_labels, num_classes=predictions.shape[1])
        pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).cpu().numpy().astype(np.uint8)

        # 各クラスのマスクを保存
        output_dir = "nifti_predictions0206_2"
        os.makedirs(output_dir, exist_ok=True)
        if not hasattr(self, "global_sample_idx"):
            self.global_sample_idx = 0

        for i in range(predictions.shape[0]):
            sample_idx = self.global_sample_idx
            self.global_sample_idx += 1

            print("今ここ！", sample_idx)

            # 1. 元の画像データを保存
            image_data = images[i].cpu().numpy().squeeze()
            image_path = os.path.join(output_dir, f"sample_{sample_idx}_image.nii.gz")
            nib.save(nib.Nifti1Image(image_data, np.eye(4)), image_path)
            print(f"Saved: {image_path}")

            # 2. Ground Truth（マスク）を保存
            gt = masks_onehot[i].cpu().numpy().astype(np.uint8)
            for class_idx in range(gt.shape[0]):
                gt_path = os.path.join(output_dir, f"sample_{sample_idx}_class_{class_idx}_gt.nii.gz")
                nib.save(nib.Nifti1Image(gt[class_idx], np.eye(4)), gt_path)
                print(f"Saved: {gt_path}")

            # 3. 予測マスクを保存
            for class_idx in range(predictions.shape[1]):
                pred_path = os.path.join(output_dir, f"sample_{sample_idx}_class_{class_idx}_pred.nii.gz")
                nib.save(nib.Nifti1Image(pred_onehot[i, class_idx], np.eye(4)), pred_path)
                print(f"Saved: {pred_path}")

        # Dice係数の計算 (argmaxによるクラスインデックスで正しく評価)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_labels,    # [B, H, W, D] クラスインデックス
            masks_labels,   # [B, H, W, D] クラスインデックス
            mode="multiclass",
            num_classes=predictions.shape[1],
        )
        # クラスごとのDice [B, C] → バッチ平均 → [C]
        dice_per_class = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").mean(dim=0)

        self.test_outputs.append({"test_dice_score": dice_per_class})
        self.log("test_dice_score", dice_per_class.mean(), on_epoch=True)

        return {"test_dice_score": dice_per_class}

    def on_test_epoch_end(self):
        if not self.test_outputs:
            print("Error: self.test_outputs is empty.")
            return

        # [N_batches, C] → [C]
        all_dice = torch.stack([x["test_dice_score"] for x in self.test_outputs])
        mean_dice = all_dice.mean(dim=0)

        print(f"nerve(class 1)  Dice: {mean_dice[1].item():.4f}")
        print(f"spinal(class 2) Dice: {mean_dice[2].item():.4f}")
        print(f"Overall Dice (nerve+spinal 平均): {mean_dice[1:].mean().item():.4f}")

        self.test_outputs = []
