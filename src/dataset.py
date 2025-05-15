# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from pycocotools.coco import COCO  # COCO API

# Assuming config.py, utils.py are in the same src directory or PYTHONPATH is set
import config
import utils


class CocoAdjustmentDataset(Dataset):
    def __init__(self, annotations_path, img_dir, for_training=True):
        super(CocoAdjustmentDataset, self).__init__()
        self.img_dir = img_dir
        self.coco = COCO(annotations_path)
        self.for_training = (
            for_training  # To control jittering etc. if needed for val/test
        )

        # Create a list of all annotations (each annotation is a sample)
        self.annotation_ids = self.coco.getAnnIds()

        # Filter out annotations that might be problematic (e.g., very small area)
        # This is an optional step, but can improve training stability.
        # For now, we'll use all annotations.
        # self.annotation_ids = [ann_id for ann_id in self.annotation_ids if self.coco.loadAnns(ann_id)[0]['area'] > some_threshold]

        # Pre-defined transformations for the image tensor
        self.normalize_transform = T.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.annotation_ids)

    def __getitem__(self, idx):
        ann_id = self.annotation_ids[idx]
        annotation = self.coco.loadAnns(ann_id)[0]
        img_info = self.coco.loadImgs(annotation["image_id"])[0]

        img_path = os.path.join(self.img_dir, img_info["file_name"])
        original_pil_image = Image.open(img_path).convert("RGB")
        img_w, img_h = original_pil_image.size

        # COCO bbox is [x_min, y_min, width, height]
        # Convert to (x_min, y_min, x_max, y_max)
        x, y, w, h = annotation["bbox"]
        gt_bbox_original_xyxy = (x, y, x + w, y + h)

        # 1. Apply Jittering to GT to get rough BBox (only if for_training)
        if self.for_training:
            rough_bbox_original_xyxy = utils.jitter_bbox(
                gt_bbox_original_xyxy,
                img_w,
                img_h,
                center_jitter_ratio=config.CENTER_JITTER_RATIO,
                scale_jitter_ratio_range=(
                    1.0 - config.SCALE_JITTER_RATIO,
                    1.0 + config.SCALE_JITTER_RATIO,
                ),
            )
        else:
            rough_bbox_original_xyxy = gt_bbox_original_xyxy

        # 2. Determine Crop Region using rough_bbox and Buffer_Ratio
        r_x_min, r_y_min, r_x_max, r_y_max = rough_bbox_original_xyxy
        r_w = r_x_max - r_x_min
        r_h = r_y_max - r_y_min
        if r_w <= 0:
            r_w = 1  # ensure width is positive
        if r_h <= 0:
            r_h = 1  # ensure height is positive

        crop_center_x = r_x_min + r_w / 2
        crop_center_y = r_y_min + r_h / 2

        buffer_ratio_to_use = config.BUFFER_RATIO

        crop_w = int(r_w * buffer_ratio_to_use)
        crop_h = int(r_h * buffer_ratio_to_use)

        crop_x_min = max(0, int(crop_center_x - crop_w / 2))
        crop_y_min = max(0, int(crop_center_y - crop_h / 2))

        crop_x_max_inclusive = min(img_w - 1, int(crop_center_x + crop_w / 2))
        crop_y_max_inclusive = min(img_h - 1, int(crop_center_y + crop_h / 2))

        # Ensure crop dimensions are valid
        if crop_x_max_inclusive <= crop_x_min:
            crop_x_max_inclusive = (
                crop_x_min + 1 if crop_x_min < img_w - 1 else crop_x_min
            )
        if crop_y_max_inclusive <= crop_y_min:
            crop_y_max_inclusive = (
                crop_y_min + 1 if crop_y_min < img_h - 1 else crop_y_min
            )

        # Final clip
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max_inclusive = min(img_w - 1, crop_x_max_inclusive)
        crop_y_max_inclusive = min(img_h - 1, crop_y_max_inclusive)

        if crop_x_min >= crop_x_max_inclusive or crop_y_min >= crop_y_max_inclusive:
            if crop_x_min >= img_w - 1:
                crop_x_min = img_w - 2
            crop_x_max_inclusive = crop_x_min + 1
            if crop_y_min >= img_h - 1:
                crop_y_min = img_h - 2
            crop_y_max_inclusive = crop_y_min + 1
            crop_x_min = max(0, crop_x_min)
            crop_y_min = max(0, crop_y_min)
            crop_x_max_inclusive = min(img_w - 1, crop_x_max_inclusive)
            crop_y_max_inclusive = min(img_h - 1, crop_y_max_inclusive)

        # 3. Crop Image (PIL.crop uses exclusive right and lower bounds)
        cropped_image_pil = original_pil_image.crop(
            (crop_x_min, crop_y_min, crop_x_max_inclusive + 1, crop_y_max_inclusive + 1)
        )

        actual_cropped_w = cropped_image_pil.width
        actual_cropped_h = cropped_image_pil.height

        if actual_cropped_w <= 0 or actual_cropped_h <= 0:
            print(
                f"Warning: Invalid crop for ann_id {ann_id}, img_id {img_info['id']}. Size: {actual_cropped_w}x{actual_cropped_h}. Cropping params: ({crop_x_min}, {crop_y_min}, {crop_x_max_inclusive + 1}, {crop_y_max_inclusive + 1})"
            )
            input_tensor = torch.zeros(
                (3, config.CROP_IMG_HEIGHT, config.CROP_IMG_WIDTH)
            )
            input_tensor = self.normalize_transform(input_tensor)
            targets = {
                "top": torch.zeros(config.CROP_IMG_HEIGHT),
                "bottom": torch.zeros(config.CROP_IMG_HEIGHT),
                "left": torch.zeros(config.CROP_IMG_WIDTH),
                "right": torch.zeros(config.CROP_IMG_WIDTH),
            }
            return input_tensor, targets

        # 4. Resize cropped image to model's expected input size and convert to tensor
        resized_image_pil = TF.resize(
            cropped_image_pil, (config.CROP_IMG_HEIGHT, config.CROP_IMG_WIDTH)
        )
        input_tensor = TF.to_tensor(resized_image_pil)
        input_tensor = self.normalize_transform(
            input_tensor
        )  # Normalize after to_tensor

        # 5. Transform GT bbox coordinates to the *resized_image_pil* coordinate system
        gt_x_min_orig, gt_y_min_orig, gt_x_max_orig, gt_y_max_orig = (
            gt_bbox_original_xyxy
        )

        gt_x_min_rel_crop = gt_x_min_orig - crop_x_min
        gt_y_min_rel_crop = gt_y_min_orig - crop_y_min
        gt_x_max_rel_crop = gt_x_max_orig - crop_x_min
        gt_y_max_rel_crop = gt_y_max_orig - crop_y_min

        scale_x = (
            config.CROP_IMG_WIDTH / actual_cropped_w if actual_cropped_w > 0 else 0
        )
        scale_y = (
            config.CROP_IMG_HEIGHT / actual_cropped_h if actual_cropped_h > 0 else 0
        )

        final_gt_left = gt_x_min_rel_crop * scale_x
        final_gt_right = gt_x_max_rel_crop * scale_x
        final_gt_top = gt_y_min_rel_crop * scale_y
        final_gt_bottom = gt_y_max_rel_crop * scale_y

        target_left = utils.get_target_vector(
            max(0, min(final_gt_left, config.CROP_IMG_WIDTH - 1)),
            config.CROP_IMG_WIDTH,
        )
        target_right = utils.get_target_vector(
            max(0, min(final_gt_right, config.CROP_IMG_WIDTH - 1)),
            config.CROP_IMG_WIDTH,
        )
        target_top = utils.get_target_vector(
            max(0, min(final_gt_top, config.CROP_IMG_HEIGHT - 1)),
            config.CROP_IMG_HEIGHT,
        )
        target_bottom = utils.get_target_vector(
            max(0, min(final_gt_bottom, config.CROP_IMG_HEIGHT - 1)),
            config.CROP_IMG_HEIGHT,
        )

        targets = {
            "top": target_top,
            "bottom": target_bottom,
            "left": target_left,
            "right": target_right,
        }
        return input_tensor, targets
