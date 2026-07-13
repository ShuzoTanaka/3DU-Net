import os
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch

def calculate_dice_for_class1(ground_truth_dir, prediction_dir):
    """
    Ground Truthと予測マスクから、クラス1（画素値127）のDiceスコアを計算し、真っ黒なGround Truthを除外して平均スコアを計算する。

    Args:
        ground_truth_dir (str): Ground Truthが保存されているディレクトリ。
        prediction_dir (str): 予測マスクが保存されているディレクトリ。
    """
    # ファイル名の取得
    gt_files = sorted(os.listdir(ground_truth_dir))
    pred_files = sorted(os.listdir(prediction_dir))

    if len(gt_files) != len(pred_files):
        print("Error: Ground Truthと予測マスクのファイル数が一致しません。")
        return

    dice_scores = []

    for gt_file, pred_file in zip(gt_files, pred_files):
        # ファイル名が一致しているか確認
        if gt_file != pred_file:
            print(f"Warning: ファイル名が一致しません。{gt_file} != {pred_file}")
            continue

        # 画像を読み込み
        gt_path = os.path.join(ground_truth_dir, gt_file)
        pred_path = os.path.join(prediction_dir, pred_file)

        ground_truth = np.array(Image.open(gt_path))  # Ground Truth
        prediction = np.array(Image.open(pred_path))  # Prediction

        # クラス1（画素値127）のバイナリマスクを作成
        gt_class1 = (ground_truth == 127).astype(np.uint8)
        pred_class1 = (prediction == 127).astype(np.uint8)

        # Ground Truthが真っ黒な場合はスキップ
        if np.sum(gt_class1) == 0:
            print(f"Skipping {gt_file}: Ground Truth is empty for class 1.")
            continue

        # Tensorに変換
        gt_class1_tensor = torch.tensor(gt_class1, dtype=torch.bool).view(1, -1)
        pred_class1_tensor = torch.tensor(pred_class1, dtype=torch.bool).view(1, -1)

        # Diceスコアの計算
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_class1_tensor,
            gt_class1_tensor,
            mode="binary",
        )
        dice_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").mean().item()

        # 結果を保存
        dice_scores.append((gt_file, dice_score))
        print(f"File: {gt_file}, Dice Score for Class 1: {dice_score:.4f}")

    # 平均Diceスコア
    if dice_scores:
        mean_dice = sum(score for _, score in dice_scores) / len(dice_scores)
        print(f"\nAverage Dice Score for Class 1: {mean_dice:.4f}")
    else:
        print("No valid samples for Dice score calculation.")


if __name__ == '__main__':
    # Ground Truthと予測マスクのディレクトリ
    ground_truth_dir = "test_predictions/true_masks"
    prediction_dir = "test_predictions/pred_masks"

    # クラス1に対するDiceスコアの計算と出力
    calculate_dice_for_class1(ground_truth_dir, prediction_dir)
