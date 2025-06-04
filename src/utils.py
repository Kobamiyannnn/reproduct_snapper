import torch
import random
import torchvision.ops.boxes as box_ops  # For box_iou if needed, or implement manually


def jitter_bbox(
    bbox_coords,
    img_width,
    img_height,
    center_jitter_ratio=0.1,
    scale_jitter_ratio_range=(0.9, 1.1),
):
    """
    Applies jittering to a bounding box.
    Args:
        bbox_coords: Tuple (x_min, y_min, x_max, y_max) of the ground truth bounding box.
        img_width: Width of the original image.
        img_height: Height of the original image.
        center_jitter_ratio: Max percentage of bbox dimension to shift the center.
        scale_jitter_ratio_range: Tuple (min_scale_factor, max_scale_factor) for rescaling.
                                  Paper states "randomly sampled ratio between 0.9 and 1.1".
    Returns:
        Tuple (x_min, y_min, x_max, y_max) of the jittered bounding box,
        clipped to image boundaries.
    """
    x_min, y_min, x_max, y_max = bbox_coords
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    # Handle zero-width/height bboxes if they occur (e.g. from bad annotations)
    if bbox_w <= 0:
        bbox_w = 1
    if bbox_h <= 0:
        bbox_h = 1

    center_x = x_min + bbox_w / 2
    center_y = y_min + bbox_h / 2

    # 1. Shift center
    dx_limit = center_jitter_ratio * bbox_w
    dy_limit = center_jitter_ratio * bbox_h
    dx = random.uniform(-dx_limit, dx_limit)
    dy = random.uniform(-dy_limit, dy_limit)
    new_center_x = center_x + dx
    new_center_y = center_y + dy

    # 2. Rescale dimensions
    rescale_factor_w = random.uniform(
        scale_jitter_ratio_range[0], scale_jitter_ratio_range[1]
    )
    rescale_factor_h = random.uniform(
        scale_jitter_ratio_range[0], scale_jitter_ratio_range[1]
    )

    new_bbox_w = bbox_w * rescale_factor_w
    new_bbox_h = bbox_h * rescale_factor_h

    # Calculate new min/max coordinates
    new_x_min = new_center_x - new_bbox_w / 2
    new_y_min = new_center_y - new_bbox_h / 2
    new_x_max = new_center_x + new_bbox_w / 2
    new_y_max = new_center_y + new_bbox_h / 2

    # 3. Clip to image boundaries
    final_x_min = max(0, int(round(new_x_min)))
    final_y_min = max(0, int(round(new_y_min)))
    final_x_max = min(img_width - 1, int(round(new_x_max)))
    final_y_max = min(img_height - 1, int(round(new_y_max)))

    # Ensure min < max and width/height > 0 after clipping
    if final_x_min >= final_x_max:
        final_x_max = final_x_min + 1
    if final_y_min >= final_y_max:
        final_y_max = final_y_min + 1

    # Final clip to ensure they are within bounds after potential +1 adjustment
    final_x_min = max(0, final_x_min)
    final_y_min = max(0, final_y_min)
    final_x_max = min(img_width - 1, final_x_max)
    final_y_max = min(img_height - 1, final_y_max)

    # One last check if min >= max after all adjustments
    if final_x_min >= final_x_max:  # Should not happen if img_width > 0
        final_x_min = final_x_max - 1 if final_x_max > 0 else 0
    if final_y_min >= final_y_max:  # Should not happen if img_height > 0
        final_y_min = final_y_max - 1 if final_y_max > 0 else 0

    return final_x_min, final_y_min, final_x_max, final_y_max


def get_target_vector(edge_coord, length, device="cpu"):
    """
    Creates a binary target vector for an edge.
    Args:
        edge_coord: The coordinate of the edge.
        length: The total length of the dimension (image height or width).
        device: The torch device to create the tensor on.
    Returns:
        A binary torch tensor of shape (length,).
    """
    target = torch.zeros(length, device=device)
    # Ensure coordinate is an integer and within bounds for indexing
    coord = int(round(edge_coord))
    if 0 <= coord < length:
        target[coord] = 1.0
    return target


def predictions_to_bboxes(predictions_logits, image_width, image_height, device="cpu"):
    """
    Converts model's edge prediction logits to bounding box coordinates (x_min, y_min, x_max, y_max).
    Args:
        predictions_logits: Dictionary from model output {"top": (B, H), "bottom": (B, H), "left": (B, W), "right": (B, W)}
        image_width: Width of the image for which predictions are made (e.g., config.CROP_IMG_WIDTH)
        image_height: Height of the image (e.g., config.CROP_IMG_HEIGHT)
        device: torch device
    Returns:
        Tensor of shape (B, 4) with (x_min, y_min, x_max, y_max) for each prediction in the batch.
    """
    pred_top_probs = torch.sigmoid(predictions_logits["top"])  # (B, H)
    pred_bottom_probs = torch.sigmoid(predictions_logits["bottom"])  # (B, H)
    pred_left_probs = torch.sigmoid(predictions_logits["left"])  # (B, W)
    pred_right_probs = torch.sigmoid(predictions_logits["right"])  # (B, W)

    pred_y_min = torch.argmax(pred_top_probs, dim=1)  # (B,)
    pred_y_max = torch.argmax(pred_bottom_probs, dim=1)  # (B,)
    pred_x_min = torch.argmax(pred_left_probs, dim=1)  # (B,)
    pred_x_max = torch.argmax(pred_right_probs, dim=1)  # (B,)

    y_min_final = torch.min(pred_y_min, pred_y_max)
    y_max_final = torch.max(pred_y_min, pred_y_max)

    x_min_final = torch.min(pred_x_min, pred_x_max)
    x_max_final = torch.max(pred_x_min, pred_x_max)

    x_max_final = torch.where(
        x_min_final == x_max_final, x_max_final + 1, x_max_final
    ).clamp(max=image_width - 1)
    y_max_final = torch.where(
        y_min_final == y_max_final, y_max_final + 1, y_max_final
    ).clamp(max=image_height - 1)

    return torch.stack(
        (x_min_final, y_min_final, x_max_final, y_max_final), dim=1
    ).float()  # (B, 4)


def get_gt_bboxes_from_targets(targets_dict, image_width, image_height, device="cpu"):
    """
    Converts target vectors (one-hot for each edge) to GT bounding box coordinates.
    Args:
        targets_dict: Dictionary from dataloader {"top": (B,H), "bottom": (B,H), "left": (B,W), "right": (B,W)}
        image_width: Width of the image (config.CROP_IMG_WIDTH)
        image_height: Height of the image (config.CROP_IMG_HEIGHT)
        device: torch device
    Returns:
        Tensor of shape (B, 4) with (x_min, y_min, x_max, y_max) for each GT in the batch.
    """
    gt_y_min = torch.argmax(targets_dict["top"].to(device), dim=1)
    gt_y_max = torch.argmax(targets_dict["bottom"].to(device), dim=1)
    gt_x_min = torch.argmax(targets_dict["left"].to(device), dim=1)
    gt_x_max = torch.argmax(targets_dict["right"].to(device), dim=1)

    return torch.stack((gt_x_min, gt_y_min, gt_x_max, gt_y_max), dim=1).float()


def calculate_iou_batch(pred_bboxes, gt_bboxes):
    """
    Calculates Intersection over Union (IoU) for a batch of predicted and ground truth bboxes.
    Args:
        pred_bboxes: Tensor (B, 4) of predicted bboxes (x1, y1, x2, y2)
        gt_bboxes: Tensor (B, 4) of ground truth bboxes (x1, y1, x2, y2)
    Returns:
        Tensor (B,) of IoU values for each pair.
    """
    xA = torch.max(pred_bboxes[:, 0], gt_bboxes[:, 0])
    yA = torch.max(pred_bboxes[:, 1], gt_bboxes[:, 1])
    xB = torch.min(pred_bboxes[:, 2], gt_bboxes[:, 2])
    yB = torch.min(pred_bboxes[:, 3], gt_bboxes[:, 3])

    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

    predBoxArea = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (
        pred_bboxes[:, 3] - pred_bboxes[:, 1]
    )
    gtBoxArea = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
        gt_bboxes[:, 3] - gt_bboxes[:, 1]
    )

    unionArea = predBoxArea + gtBoxArea - interArea

    iou = interArea / (unionArea + 1e-6)
    return iou


def calculate_edge_deviance_batch(pred_bboxes, gt_bboxes, thresholds=(1.0, 3.0)):
    """
    Calculates Edge Deviance for a batch.
    Args:
        pred_bboxes: Tensor (B, 4) of predicted bboxes (x_min, y_min, x_max, y_max)
        gt_bboxes: Tensor (B, 4) of ground truth bboxes (x_min, y_min, x_max, y_max)
        thresholds: Tuple of pixel thresholds (e.g., (1.0, 3.0))
    Returns:
        A dictionary containing:
            'mean_dev_left', 'mean_dev_top', 'mean_dev_right', 'mean_dev_bottom':
                Mean absolute pixel deviance for each edge (left, top, right, bottom respectively).
                Lower values indicate more accurate edge placement.
            'frac_dev_left_le_Xpx', 'frac_dev_top_le_Xpx', 'frac_dev_right_le_Xpx', 'frac_dev_bottom_le_Xpx':
                Fraction of samples where the respective edge's pixel deviance is less than or equal to X pixels (e.g., 1px, 3px).
                Higher values (closer to 1.0) indicate better precision for that edge at the given threshold.
    """
    dev_left = torch.abs(pred_bboxes[:, 0] - gt_bboxes[:, 0])
    dev_top = torch.abs(pred_bboxes[:, 1] - gt_bboxes[:, 1])
    dev_right = torch.abs(pred_bboxes[:, 2] - gt_bboxes[:, 2])
    dev_bottom = torch.abs(pred_bboxes[:, 3] - gt_bboxes[:, 3])

    results = {
        "mean_dev_left": torch.mean(dev_left).item(),
        "mean_dev_top": torch.mean(dev_top).item(),
        "mean_dev_right": torch.mean(dev_right).item(),
        "mean_dev_bottom": torch.mean(dev_bottom).item(),
    }

    for i, thresh in enumerate(thresholds):
        thresh_str = str(int(thresh)) if thresh.is_integer() else str(thresh)
        results[f"frac_dev_left_le_{thresh_str}px"] = torch.sum(
            dev_left <= thresh
        ).item() / len(dev_left)
        results[f"frac_dev_top_le_{thresh_str}px"] = torch.sum(
            dev_top <= thresh
        ).item() / len(dev_top)
        results[f"frac_dev_right_le_{thresh_str}px"] = torch.sum(
            dev_right <= thresh
        ).item() / len(dev_right)
        results[f"frac_dev_bottom_le_{thresh_str}px"] = torch.sum(
            dev_bottom <= thresh
        ).item() / len(dev_bottom)

    return results


def calculate_corner_deviance_batch(pred_bboxes, gt_bboxes, thresholds=(1.0, 3.0)):
    """
    Calculates Corner Deviance for a batch.
    Args:
        pred_bboxes: Tensor (B, 4) of predicted bboxes (x_min, y_min, x_max, y_max)
        gt_bboxes: Tensor (B, 4) of ground truth bboxes (x_min, y_min, x_max, y_max)
        thresholds: Tuple of pixel thresholds.
    Returns:
        A dictionary containing:
            'mean_dev_tl', 'mean_dev_tr', 'mean_dev_bl', 'mean_dev_br':
                Mean L1 (Manhattan) distance in pixels for each corner (tl: top-left, tr: top-right, bl: bottom-left, br: bottom-right).
                L1 distance = |pred_x - gt_x| + |pred_y - gt_y|.
                Lower values indicate more accurate corner placement.
            'frac_dev_tl_le_Xpx', 'frac_dev_tr_le_Xpx', 'frac_dev_bl_le_Xpx', 'frac_dev_br_le_Xpx':
                Fraction of samples where the respective corner's L1 distance is less than or equal to X pixels (e.g., 1px, 3px).
                Higher values (closer to 1.0) indicate better precision for that corner at the given threshold.
    """
    dev_tl_x = torch.abs(pred_bboxes[:, 0] - gt_bboxes[:, 0])
    dev_tl_y = torch.abs(pred_bboxes[:, 1] - gt_bboxes[:, 1])
    dist_tl = dev_tl_x + dev_tl_y

    dev_tr_x = torch.abs(pred_bboxes[:, 2] - gt_bboxes[:, 2])
    dev_tr_y = torch.abs(pred_bboxes[:, 1] - gt_bboxes[:, 1])
    dist_tr = dev_tr_x + dev_tr_y

    dev_bl_x = torch.abs(pred_bboxes[:, 0] - gt_bboxes[:, 0])
    dev_bl_y = torch.abs(pred_bboxes[:, 3] - gt_bboxes[:, 3])
    dist_bl = dev_bl_x + dev_bl_y

    dev_br_x = torch.abs(pred_bboxes[:, 2] - gt_bboxes[:, 2])
    dev_br_y = torch.abs(pred_bboxes[:, 3] - gt_bboxes[:, 3])
    dist_br = dev_br_x + dev_br_y

    results = {
        "mean_dev_tl": torch.mean(dist_tl).item(),
        "mean_dev_tr": torch.mean(dist_tr).item(),
        "mean_dev_bl": torch.mean(dist_bl).item(),
        "mean_dev_br": torch.mean(dist_br).item(),
    }

    for i, thresh in enumerate(thresholds):
        thresh_str = str(int(thresh)) if thresh.is_integer() else str(thresh)
        results[f"frac_dev_tl_le_{thresh_str}px"] = torch.sum(
            dist_tl <= thresh
        ).item() / len(dist_tl)
        results[f"frac_dev_tr_le_{thresh_str}px"] = torch.sum(
            dist_tr <= thresh
        ).item() / len(dist_tr)
        results[f"frac_dev_bl_le_{thresh_str}px"] = torch.sum(
            dist_bl <= thresh
        ).item() / len(dist_bl)
        results[f"frac_dev_br_le_{thresh_str}px"] = torch.sum(
            dist_br <= thresh
        ).item() / len(dist_br)

    return results


# You might add other utility functions here later, e.g., for IoU calculation
