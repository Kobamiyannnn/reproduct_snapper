import torch
import random
# import torchvision.ops.boxes as box_ops  # For box_iou if needed, or implement manually


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
    # TODO: "rescales the dimensions of the bounding box by a randomly sampled ratio between 0.9 and 1.1"という記述があるが、
    # `rescale_factor_w`と`rescale_factor_h`が異なっており、原論文より厳しい前処理設計となっている可能性がある。
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


def calculate_edge_accuracy_batch(pred_bboxes, gt_bboxes, thresholds=(1.0, 3.0)):
    """
    Calculates the percentage of edges within certain pixel deviation thresholds.
    This metric aligns with the "Edge < Xpx" metric from the Snapper paper.

    For each bounding box, it calculates the proportion of its 4 edges that are
    within the given deviation threshold. The final metric is the average of these
    proportions over the entire batch.

    Args:
        pred_bboxes: Tensor (B, 4) of predicted bboxes (x_min, y_min, x_max, y_max).
        gt_bboxes: Tensor (B, 4) of ground truth bboxes (x_min, y_min, x_max, y_max).
        thresholds: A tuple of pixel deviation thresholds.

    Returns:
        A dictionary with keys like 'Edge < 1px', 'Edge < 3px', etc., where values
        are the batch-averaged percentage of edges meeting the threshold.
    """
    if pred_bboxes.shape[0] == 0:
        return {f"Edge < {t}px": 0.0 for t in thresholds}

    # Calculate absolute deviations for each of the 4 edges
    dev_left = torch.abs(pred_bboxes[:, 0] - gt_bboxes[:, 0])
    dev_top = torch.abs(pred_bboxes[:, 1] - gt_bboxes[:, 1])
    dev_right = torch.abs(pred_bboxes[:, 2] - gt_bboxes[:, 2])
    dev_bottom = torch.abs(pred_bboxes[:, 3] - gt_bboxes[:, 3])

    # Stack deviations for easier processing: (B, 4)
    edge_deviations = torch.stack([dev_left, dev_top, dev_right, dev_bottom], dim=1)

    results = {}
    for thresh in thresholds:
        # Check which edges are within the threshold: (B, 4) boolean tensor
        within_thresh = edge_deviations <= thresh
        # Calculate proportion of correct edges per box: (B,)
        proportion_per_box = torch.mean(within_thresh.float(), dim=1)
        # Average the proportions over the batch
        batch_accuracy = torch.mean(proportion_per_box).item()
        results[f"Edge < {int(thresh)}px"] = batch_accuracy

    return results


def calculate_corner_accuracy_batch(pred_bboxes, gt_bboxes, thresholds=(1.0, 3.0)):
    """
    Calculates the percentage of corners within certain pixel deviation thresholds.
    This metric aligns with the "Corner < Xpx" metric from the Snapper paper.

    For each bounding box, it calculates the proportion of its 4 corners that are
    within the given Euclidean distance threshold. The final metric is the average of
    these proportions over the entire batch.

    Args:
        pred_bboxes: Tensor (B, 4) of predicted bboxes (x_min, y_min, x_max, y_max).
        gt_bboxes: Tensor (B, 4) of ground truth bboxes (x_min, y_min, x_max, y_max).
        thresholds: A tuple of pixel deviation thresholds.

    Returns:
        A dictionary with keys like 'Corner < 1px', 'Corner < 3px', etc., where values
        are the batch-averaged percentage of corners meeting the threshold.
    """
    if pred_bboxes.shape[0] == 0:
        return {f"Corner < {t}px": 0.0 for t in thresholds}

    # Extract corner coordinates
    pred_corners = torch.stack(
        [
            pred_bboxes[:, 0],
            pred_bboxes[:, 1],  # Top-left
            pred_bboxes[:, 2],
            pred_bboxes[:, 1],  # Top-right
            pred_bboxes[:, 0],
            pred_bboxes[:, 3],  # Bottom-left
            pred_bboxes[:, 2],
            pred_bboxes[:, 3],  # Bottom-right
        ],
        dim=1,
    ).view(-1, 4, 2)  # (B, 4, 2)

    gt_corners = torch.stack(
        [
            gt_bboxes[:, 0],
            gt_bboxes[:, 1],
            gt_bboxes[:, 2],
            gt_bboxes[:, 1],
            gt_bboxes[:, 0],
            gt_bboxes[:, 3],
            gt_bboxes[:, 2],
            gt_bboxes[:, 3],
        ],
        dim=1,
    ).view(-1, 4, 2)  # (B, 4, 2)

    # Calculate Euclidean distance for each corner: (B, 4)
    corner_deviations = torch.sqrt(torch.sum((pred_corners - gt_corners) ** 2, dim=2))

    results = {}
    for thresh in thresholds:
        # Check which corners are within the threshold: (B, 4) boolean tensor
        within_thresh = corner_deviations <= thresh
        # Calculate proportion of correct corners per box: (B,)
        proportion_per_box = torch.mean(within_thresh.float(), dim=1)
        # Average the proportions over the batch
        batch_accuracy = torch.mean(proportion_per_box).item()
        results[f"Corner < {int(thresh)}px"] = batch_accuracy

    return results


def compute_metrics(
    pred_bboxes, gt_bboxes, iou_threshold=0.9, deviance_thresholds=(1.0, 3.0)
):
    """
    Computes a set of evaluation metrics for a batch of predictions.
    Args:
        pred_bboxes: Tensor (B, 4) of predicted bboxes (x_min, y_min, x_max, y_max).
        gt_bboxes: Tensor (B, 4) of ground truth bboxes (x_min, y_min, x_max, y_max).
        iou_threshold: The IoU threshold to consider a prediction "good".
        deviance_thresholds: Tuple of pixel thresholds for edge/corner accuracy.
    Returns:
        A dictionary containing all computed metrics.
    """
    if pred_bboxes.shape[0] == 0:
        metrics = {
            "mIoU": 0.0,
            f"IoU > {iou_threshold}": 0.0,
        }
        edge_acc = calculate_edge_accuracy_batch(
            pred_bboxes, gt_bboxes, deviance_thresholds
        )
        corner_acc = calculate_corner_accuracy_batch(
            pred_bboxes, gt_bboxes, deviance_thresholds
        )
        metrics.update(edge_acc)
        metrics.update(corner_acc)
        return metrics

    iou_values = calculate_iou_batch(pred_bboxes, gt_bboxes)

    metrics = {
        "mIoU": torch.mean(iou_values).item(),
        f"IoU > {iou_threshold}": torch.mean(
            (iou_values > iou_threshold).float()
        ).item(),
    }

    edge_accuracy = calculate_edge_accuracy_batch(
        pred_bboxes, gt_bboxes, deviance_thresholds
    )
    metrics.update(edge_accuracy)

    corner_accuracy = calculate_corner_accuracy_batch(
        pred_bboxes, gt_bboxes, deviance_thresholds
    )
    metrics.update(corner_accuracy)

    return metrics


# You might add other utility functions here later, e.g., for IoU calculation
