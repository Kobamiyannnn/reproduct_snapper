import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize

from models import BoundingBoxAdjustmentModel
from utils import predictions_to_bboxes
import config as cfg  # デフォルト設定

class ConfigModelInit:  # Renamed to avoid conflict if any global Config exists
        IMAGE_SIZE = 224

def load_model(model_path, device):
    """学習済みモデルをロードする"""
    # configからモデルパラメータを読み込む想定だが、現状モデル保存時に構造も保存しているため、
    # 直接ロードできる。より柔軟にするにはconfigからモデル構造を決定する。
    model = BoundingBoxAdjustmentModel(
        config=ConfigModelInit(),
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, rough_bbox_xywh, buffer_ratio, img_size, device):
    """
    入力画像を前処理する。
    1. 画像を読み込む。
    2. rough_bbox_coordsとbuffer_ratioに基づいてクロップ領域を計算。
    3. 画像をクロップ。
    4. img_sizeにリサイズ。
    5. テンソルに変換し、正規化。
    6. クロップ情報（元画像におけるクロップ領域、リサイズスケールなど）を返す。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]

    # rough_bbox_xywh を [xmin, ymin, xmax, ymax] に変換
    x_min_rough, y_min_rough, width_rough, height_rough = rough_bbox_xywh
    rough_bbox_coords = [
        x_min_rough,
        y_min_rough,
        x_min_rough + width_rough,
        y_min_rough + height_rough,
    ]
    xmin, ymin, xmax, ymax = rough_bbox_coords
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin

    # バッファを適用してクロップ領域を計算
    crop_w = bbox_w * buffer_ratio
    crop_h = bbox_h * buffer_ratio

    center_x = xmin + bbox_w / 2
    center_y = ymin + bbox_h / 2

    crop_xmin = int(center_x - crop_w / 2)
    crop_ymin = int(center_y - crop_h / 2)
    crop_xmax = int(center_x + crop_w / 2)
    crop_ymax = int(center_y + crop_h / 2)

    # 元画像境界でのクリッピング
    crop_xmin_clipped = max(0, crop_xmin)
    crop_ymin_clipped = max(0, crop_ymin)
    crop_xmax_clipped = min(original_w, crop_xmax)
    crop_ymax_clipped = min(original_h, crop_ymax)

    # クロップ実行
    cropped_image = image[
        crop_ymin_clipped:crop_ymax_clipped, crop_xmin_clipped:crop_xmax_clipped
    ]
    cropped_h, cropped_w = cropped_image.shape[:2]

    if cropped_h == 0 or cropped_w == 0:
        raise ValueError(
            "Cropped image has zero height or width. Check bbox and buffer."
        )

    # リサイズ
    resized_image = cv2.resize(cropped_image, (img_size, img_size))

    # テンソルに変換して正規化
    img_tensor = TF.to_tensor(resized_image)
    # config.py から MEAN と STD を読み込む必要がある
    normalize = Normalize(mean=cfg.MEAN, std=cfg.STD)
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # バッチ次元追加

    # クロップ情報: 元画像におけるクロップ領域の(xmin, ymin)と、リサイズによるスケール
    # スケールは、(元クロップ後の幅 / リサイズ後の幅, 元クロップ後の高さ / リサイズ後の高さ)
    # ただし、予測はリサイズ後の座標で行われるため、リサイズ後座標から元画像座標への変換情報が必要
    # crop_info = {
    #     "crop_box_original_normalized": [ # 元画像サイズで正規化されたクロップ領域
    #         crop_xmin_clipped / original_w,
    #         crop_ymin_clipped / original_h,
    #         crop_xmax_clipped / original_w,
    #         crop_ymax_clipped / original_h
    #     ],
    #     "crop_box_pixel_original": [ # 元画像のピクセル単位でのクロップ領域
    #         crop_xmin_clipped,
    #         crop_ymin_clipped,
    #         crop_xmax_clipped,
    #         crop_ymax_clipped
    #     ],
    #     "resize_scale_w": cropped_w / img_size, # 予測されたx座標に乗算してクロップ画像座標に戻す
    #     "resize_scale_h": cropped_h / img_size  # 予測されたy座標に乗算してクロップ画像座標に戻す
    # }

    # 予測された座標 (0-IMG_SIZE) を元画像の座標系に戻すための情報
    # pred_coord * scale + offset
    # x_pred_original = (x_pred_on_resized_patch * (cropped_w / IMG_SIZE)) + crop_xmin_clipped
    # y_pred_original = (y_pred_on_resized_patch * (cropped_h / IMG_SIZE)) + crop_ymin_clipped
    transform_info = {
        "original_image_shape": (original_h, original_w),
        "crop_origin_x": crop_xmin_clipped,  # クロップ領域の元画像における左上のx座標
        "crop_origin_y": crop_ymin_clipped,  # クロップ領域の元画像における左上のy座標
        "scale_x_to_cropped": cropped_w
        / img_size,  # リサイズ画像上のx座標 -> クロップ画像(リサイズ前)上のx座標
        "scale_y_to_cropped": cropped_h
        / img_size,  # リサイズ画像上のy座標 -> クロップ画像(リサイズ前)上のy座標
    }

    return img_tensor, transform_info


def postprocess_predictions(predictions_logits, transform_info, img_size):
    """
    モデルの出力を元画像座標系のバウンディングボックスに変換する。
    predictions_logits: (1, 4, img_size) のテンソル
    transform_info: preprocess_imageから返される座標変換情報
    img_size: モデルの入力画像サイズ
    """
    # ロジットから予測された座標 (リサイズされたクロップ画像上の座標 0-img_size) を取得
    # predictions_to_bboxes は (N, 4, img_size) or (N, 4, H, W) を期待するが、
    # 今のモデルは (N, 4, img_size) を出力するので、そのまま使える
    # predictions_to_bboxes は各辺の最も確率の高いインデックスを返す
    pred_bboxes_on_resized_patch = predictions_to_bboxes(
        predictions_logits,
        image_width=img_size,
        image_height=img_size,
        device="cuda",  # ロジットと同じデバイスを使用
    )  # (N, 4)
    pred_bbox_on_resized_patch = (
        pred_bboxes_on_resized_patch.squeeze(0).cpu().numpy()
    )  # (4,) [xmin, ymin, xmax, ymax]

    # リサイズされたクロップ画像上の座標 -> 元の(クリップされた)クロップ画像上の座標
    xmin_cropped = pred_bbox_on_resized_patch[0] * transform_info["scale_x_to_cropped"]
    ymin_cropped = pred_bbox_on_resized_patch[1] * transform_info["scale_y_to_cropped"]
    xmax_cropped = pred_bbox_on_resized_patch[2] * transform_info["scale_x_to_cropped"]
    ymax_cropped = pred_bbox_on_resized_patch[3] * transform_info["scale_y_to_cropped"]

    # クロップ画像上の座標 -> 元画像上の座標
    xmin_original = xmin_cropped + transform_info["crop_origin_x"]
    ymin_original = ymin_cropped + transform_info["crop_origin_y"]
    xmax_original = xmax_cropped + transform_info["crop_origin_x"]
    ymax_original = ymax_cropped + transform_info["crop_origin_y"]

    # 元画像の境界でクリッピング (念のため)
    original_h, original_w = transform_info["original_image_shape"]
    xmin_original = np.clip(xmin_original, 0, original_w)
    ymin_original = np.clip(ymin_original, 0, original_h)
    xmax_original = np.clip(xmax_original, 0, original_w)
    ymax_original = np.clip(ymax_original, 0, original_h)

    return [
        int(c) for c in [xmin_original, ymin_original, xmax_original, ymax_original]
    ]


def visualize_results(
    image_path, rough_bbox_xywh, adjusted_bbox_xyxy, output_path=None
):
    """結果を可視化する"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path} for visualization.")
        return

    # ラフなBBox (xywh) を (xyxy) に変換して緑色で描画
    r_x, r_y, r_w, r_h = rough_bbox_xywh
    cv2.rectangle(
        image,
        (r_x, r_y),
        (r_x + r_w, r_y + r_h),
        (0, 255, 0),  # Green
        1,
    )
    cv2.putText(
        image,
        "Rough (Green)",
        (r_x, r_y - 10 if r_y - 10 > 10 else r_y + r_h + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # 調整後のBBoxを赤色で描画 (これは元々xyxy形式で渡される想定)
    a_x1, a_y1, a_x2, a_y2 = adjusted_bbox_xyxy
    cv2.rectangle(
        image,
        (a_x1, a_y1),
        (a_x2, a_y2),
        (0, 0, 255),  # Red
        1,
    )
    cv2.putText(
        image,
        "Adjusted (Red)",
        (a_x1, a_y1 - 10 if a_y1 - 10 > 10 else a_y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")
    else:
        # ウィンドウのサイズを調整して表示
        # 元画像のサイズによっては大きすぎる場合があるため
        max_display_dim = 800
        h, w = image.shape[:2]
        if h > max_display_dim or w > max_display_dim:
            scale = max_display_dim / max(h, w)
            display_w, display_h = int(w * scale), int(h * scale)
            image_display = cv2.resize(image, (display_w, display_h))
        else:
            image_display = image

        cv2.imshow("Rough vs Adjusted BBox", image_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Bounding Box Adjustment Model Inference"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Rough bounding box coordinates as 'x_min,y_min,width,height'.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output image. If None, displays the image.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the config file (e.g., src/config.py). Uses default if None.",
    )
    # BUFFER_RATIO と IMG_SIZE は config から読むが、コマンドラインからも上書きできるようにしても良いかもしれない

    args = parser.parse_args()

    if args.config_path:
        # config_path が指定された場合のロード処理を実装 (現状はデフォルトcfgを使用)
        # 例: import importlib.util
        #      spec = importlib.util.spec_from_file_location("custom_config", args.config_path)
        #      custom_cfg = importlib.util.module_from_spec(spec)
        #      spec.loader.exec_module(custom_cfg)
        #      cfg_module = custom_cfg
        print(
            f"Using config from {args.config_path} is not yet fully implemented. Using default config."
        )
        cfg_module = cfg  # ここでは簡単のためデフォルトのcfgを使用
    else:
        cfg_module = cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        rough_bbox_xywh = [int(c) for c in args.bbox.split(",")]
        if len(rough_bbox_xywh) != 4:
            raise ValueError(
                "Bounding box must have 4 coordinates: x_min,y_min,width,height."
            )
    except ValueError as e:
        print(
            f"Error parsing bbox coordinates: {e}. Please use 'x_min,y_min,width,height' format."
        )
        return

    # 1. モデルのロード
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device)

    # 2. 画像の前処理
    print(f"Preprocessing image {args.image_path}...")
    try:
        img_tensor, transform_info = preprocess_image(
            args.image_path,
            rough_bbox_xywh,  # xywh 形式で渡す
            cfg_module.BUFFER_RATIO,  # config から取得
            cfg_module.IMG_SIZE,  # config から取得
            device,
        )
    except FileNotFoundError as e:
        print(e)
        return
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        return

    # 3. 推論の実行
    print("Running inference...")
    with torch.no_grad():
        predictions_logits = model(img_tensor)  # モデル入力は (1, C, H, W)

    # 4. 結果のポストプロセス
    print("Postprocessing predictions...")
    adjusted_bbox_coords = postprocess_predictions(
        predictions_logits,
        transform_info,
        cfg_module.IMG_SIZE,  # config から取得
    )
    print(f"Rough BBox: {rough_bbox_xywh}")
    print(f"Adjusted BBox: {adjusted_bbox_coords}")

    # 5. 可視化
    print("Visualizing results...")
    visualize_results(
        args.image_path, rough_bbox_xywh, adjusted_bbox_coords, args.output_path
    )


if __name__ == "__main__":
    main()
