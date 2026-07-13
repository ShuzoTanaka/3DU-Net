import os
import torch
import numpy as np
from PIL import Image
import nibabel as nib
from torchvision import transforms
from model_2D import MultiClassModel

# ---- 設定 ----
image_dir = "C:/Users/orilab/Desktop/Tanaka/pytorchLightning/test_data"  # 予測したい画像フォルダ
output_dir = "inference_output"  # 出力フォルダ
checkpoint_path = "C:/Users/orilab/Desktop/Tanaka/pytorchLightning/checkpoints/best-2D-2025-06-17_19-37-25-epoch=66-val_loss=0.10.ckpt"  # 実際のファイル名に置き換えてください
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 出力ディレクトリ作成
os.makedirs(output_dir, exist_ok=True)


# モデルの読み込み（修正済）
model = MultiClassModel.load_from_checkpoint(
    checkpoint_path,
    in_channels=1,
    num_classes=3,
    encoder_name="efficientnet-b0",
    map_location=device,
)
model.to(device)
model.eval()

# ---- 前処理 ----
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # (H, W) → [1, H, W]
    ]
)

# ---- 予測ループ ----
for filename in sorted(os.listdir(image_dir)):
    if not filename.lower().endswith(".png"):
        continue

    # 入力画像の読み込みと前処理
    img_path = os.path.join(image_dir, filename)
    image = Image.open(img_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 1, H, W]

    # 推論
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)  # multiclassなのでsoftmax
        preds = torch.argmax(probs, dim=1)  # [1, H, W]

    pred_mask = preds.squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W]

    # PNG保存
    pred_img = Image.fromarray(pred_mask * int(255 // (num_classes - 1)))
    pred_img.save(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pred.png"))

    # NIfTI保存（任意）
    nii_img = nib.Nifti1Image(pred_mask, affine=np.eye(4))
    nib.save(
        nii_img,
        os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pred.nii.gz"),
    )

    print(f"{filename} → 推論完了")

print("✅ すべての推論が完了しました。")
