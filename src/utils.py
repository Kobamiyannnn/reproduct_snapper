import torch
import random


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


# You might add other utility functions here later, e.g., for IoU calculation
