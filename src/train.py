# src/train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import torch.multiprocessing as mp

# Assuming config.py, models.py, dataset.py, utils.py are in the same src directory
# or PYTHONPATH is correctly set.
import config
from models import BoundingBoxAdjustmentModel
from dataset import CocoAdjustmentDataset
import utils  # Ensure utils is imported


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        target_top = targets["top"].to(device)
        target_bottom = targets["bottom"].to(device)
        target_left = targets["left"].to(device)
        target_right = targets["right"].to(device)

        optimizer.zero_grad()
        predictions = model(images)  # These are logits

        loss_t = criterion(predictions["top"], target_top)
        loss_b = criterion(predictions["bottom"], target_bottom)
        loss_l = criterion(predictions["left"], target_left)
        loss_r = criterion(predictions["right"], target_right)

        total_loss = loss_t + loss_b + loss_l + loss_r

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

        if (i + 1) % 100 == 0:  # Log every 100 batches
            print(
                f"    Batch [{i + 1}/{len(dataloader)}], Batch Loss: {total_loss.item():.4f}"
            )

    return epoch_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device, image_width, image_height):
    model.eval()
    epoch_loss = 0.0

    all_iou_scores = []
    total_samples_for_dev = 0
    sum_edge_dev_metrics = {}
    sum_corner_dev_metrics = {}

    with torch.no_grad():
        for i, (images, targets_dict) in enumerate(dataloader):
            images = images.to(device)

            predictions_logits = model(images)

            loss_t = criterion(
                predictions_logits["top"], targets_dict["top"].to(device)
            )
            loss_b = criterion(
                predictions_logits["bottom"], targets_dict["bottom"].to(device)
            )
            loss_l = criterion(
                predictions_logits["left"], targets_dict["left"].to(device)
            )
            loss_r = criterion(
                predictions_logits["right"], targets_dict["right"].to(device)
            )
            total_loss = loss_t + loss_b + loss_l + loss_r
            epoch_loss += total_loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"    Batch [{i + 1}/{len(dataloader)}], Val Batch Loss: {total_loss.item():.4f}"
                )

            pred_bboxes_batch = utils.predictions_to_bboxes(
                predictions_logits, image_width, image_height, device
            )
            gt_bboxes_batch = utils.get_gt_bboxes_from_targets(
                targets_dict, image_width, image_height, device
            )

            iou_batch = utils.calculate_iou_batch(pred_bboxes_batch, gt_bboxes_batch)
            all_iou_scores.extend(iou_batch.cpu().tolist())

            edge_dev_batch_results = utils.calculate_edge_deviance_batch(
                pred_bboxes_batch, gt_bboxes_batch
            )
            for key, value in edge_dev_batch_results.items():
                if "frac" in key:
                    num_within_thresh = value * pred_bboxes_batch.size(0)
                    sum_edge_dev_metrics[key] = (
                        sum_edge_dev_metrics.get(key, 0) + num_within_thresh
                    )
                else:
                    sum_edge_dev_metrics[key] = sum_edge_dev_metrics.get(
                        key, 0
                    ) + value * pred_bboxes_batch.size(0)

            corner_dev_batch_results = utils.calculate_corner_deviance_batch(
                pred_bboxes_batch, gt_bboxes_batch
            )
            for key, value in corner_dev_batch_results.items():
                if "frac" in key:
                    num_within_thresh = value * pred_bboxes_batch.size(0)
                    sum_corner_dev_metrics[key] = (
                        sum_corner_dev_metrics.get(key, 0) + num_within_thresh
                    )
                else:
                    sum_corner_dev_metrics[key] = sum_corner_dev_metrics.get(
                        key, 0
                    ) + value * pred_bboxes_batch.size(0)

            total_samples_for_dev += pred_bboxes_batch.size(0)

    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_iou = np.mean(all_iou_scores) if all_iou_scores else 0.0

    print("\n    --- Validation Summary ---")
    print(f"    Average Validation Loss: {avg_epoch_loss:.4f}")
    print(f"    Average IoU: {avg_iou:.4f}")

    print("    Edge Deviance:")
    if total_samples_for_dev > 0:
        for key, summed_value in sum_edge_dev_metrics.items():
            if "frac" in key:
                metric_val = summed_value / total_samples_for_dev
                print(f"        {key}: {metric_val:.4f}")
            else:
                metric_val = summed_value / total_samples_for_dev
                print(f"        {key}: {metric_val:.4f} px")

    print("    Corner Deviance:")
    if total_samples_for_dev > 0:
        for key, summed_value in sum_corner_dev_metrics.items():
            if "frac" in key:
                metric_val = summed_value / total_samples_for_dev
                print(f"        {key}: {metric_val:.4f}")
            else:
                metric_val = summed_value / total_samples_for_dev
                print(f"        {key}: {metric_val:.4f} px")
    print("    ------------------------")

    return avg_epoch_loss, avg_iou


def main():
    if config.DEVICE == "cuda":
        try:
            current_start_method = mp.get_start_method(allow_none=True)
            if current_start_method is None or current_start_method == "fork":
                mp.set_start_method("spawn", force=True)
                print("Set multiprocessing start method to 'spawn' for CUDA.")
        except RuntimeError as e:
            print(
                f"Warning: Could not set multiprocessing start method to 'spawn': {e}"
            )

    print("Starting training process...")
    seed_everything(config.RANDOM_SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    if (
        not hasattr(config, "COCO_ANNOTATIONS_PATH_TRAIN")
        or not hasattr(config, "COCO_IMG_DIR_TRAIN")
        or not config.COCO_ANNOTATIONS_PATH_TRAIN
        or not config.COCO_IMG_DIR_TRAIN
    ):
        print(
            "Error: COCO_ANNOTATIONS_PATH_TRAIN or COCO_IMG_DIR_TRAIN is not set or empty in src/config.py."
        )
        print(
            "Please set the correct paths to your COCO training data in src/config.py."
        )
        return

    coco_annotations_path_train = config.COCO_ANNOTATIONS_PATH_TRAIN
    coco_img_dir_train = config.COCO_IMG_DIR_TRAIN

    print(
        f"Loading training dataset from: {coco_img_dir_train} with annotations: {coco_annotations_path_train}"
    )
    train_dataset = CocoAdjustmentDataset(
        annotations_path=coco_annotations_path_train,
        img_dir=coco_img_dir_train,
        for_training=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )
    print(f"Training dataset loaded: {len(train_dataset)} samples.")

    # --- Validation DataLoader ---
    if (
        hasattr(config, "COCO_ANNOTATIONS_PATH_VAL")
        and hasattr(config, "COCO_IMG_DIR_VAL")
        and config.COCO_ANNOTATIONS_PATH_VAL
        and config.COCO_IMG_DIR_VAL
    ):
        coco_annotations_path_val = config.COCO_ANNOTATIONS_PATH_VAL
        coco_img_dir_val = config.COCO_IMG_DIR_VAL
        print(
            f"Loading validation dataset from: {coco_img_dir_val} with annotations: {coco_annotations_path_val}"
        )
        val_dataset = CocoAdjustmentDataset(
            annotations_path=coco_annotations_path_val,
            img_dir=coco_img_dir_val,
            for_training=False,  # Important: jittering should be off or different for validation
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if device.type == "cuda" else False,
        )
        print(f"Validation dataset loaded: {len(val_dataset)} samples.")
        use_validation = True
    else:
        print(
            "Validation data paths not found in config or are empty. Skipping validation."
        )
        val_dataloader = None
        use_validation = False

    # --- Model, Loss, Optimizer ---
    model = BoundingBoxAdjustmentModel(
        image_height=config.CROP_IMG_HEIGHT,
        image_width=config.CROP_IMG_WIDTH,
    ).to(device)
    print("Model initialized.")

    if config.LOSS_FN_TYPE == "BCEWithLogitsLoss":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss function type: {config.LOSS_FN_TYPE}")
    print(f"Using loss function: {config.LOSS_FN_TYPE}")

    if config.OPTIMIZER_TYPE == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER_TYPE == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.OPTIMIZER_TYPE}")
    print(f"Using optimizer: {config.OPTIMIZER_TYPE} with LR: {config.LEARNING_RATE}")

    # --- Checkpoint Directory ---
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {config.CHECKPOINT_DIR}")

    # --- Training Loop ---
    print(f"Starting training for {config.EPOCHS} epochs...")
    best_val_iou = float("-inf")

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")

        avg_train_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device
        )
        print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.4f}")

        avg_val_loss = None
        avg_val_iou = None

        if use_validation and val_dataloader:
            avg_val_loss, avg_val_iou = validate_one_epoch(
                model,
                val_dataloader,
                criterion,
                device,
                config.CROP_IMG_WIDTH,
                config.CROP_IMG_HEIGHT,
            )

        if use_validation and avg_val_iou is not None and avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model_iou.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                    "iou": avg_val_iou,
                },
                checkpoint_path,
            )
            print(
                f"Saved best model checkpoint (Val IoU: {best_val_iou:.4f}) to {checkpoint_path}"
            )

        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, f"model_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss if avg_val_loss is not None else -1,
                    "val_iou": avg_val_iou if avg_val_iou is not None else -1,
                },
                checkpoint_path,
            )
            print(f"Saved periodic model checkpoint to {checkpoint_path}")

    print("\nTraining finished.")


if __name__ == "__main__":
    # This allows running train.py directly as a script
    # Before running, ensure you have COCO dataset downloaded and paths in config.py are correct.
    # Also, run `pip install pycocotools` (or the windows equivalent).

    # For a quick test without actual COCO data, you might need to create
    # dummy annotation files and image directories, or modify CocoAdjustmentDataset
    # to work with some placeholder data.
    # The current script will try to load COCO data based on paths.

    main()
