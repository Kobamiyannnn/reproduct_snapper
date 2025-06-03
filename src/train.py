# src/train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim.lr_scheduler as lr_scheduler

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


def train_one_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    epoch_loss = 0.0

    scaler = torch.amp.GradScaler(device=device)

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        target_top = targets["top"].to(device)
        target_bottom = targets["bottom"].to(device)
        target_left = targets["left"].to(device)
        target_right = targets["right"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            preds_top, preds_bottom, preds_left, preds_right = model(images)

            loss_top = criterion(preds_top, target_top.float())
            loss_bottom = criterion(preds_bottom, target_bottom.float())
            loss_left = criterion(preds_left, target_left.float())
            loss_right = criterion(preds_right, target_right.float())

            total_loss = loss_top + loss_bottom + loss_left + loss_right

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_loss = total_loss.item()
        epoch_loss += current_loss

        if writer:
            writer.add_scalar(
                "Loss/train_batch", current_loss, epoch * len(dataloader) + i
            )

        if (i + 1) % 100 == 0:  # Log every 100 batches
            print(
                f"    Batch [{i + 1}/{len(dataloader)}], Batch Loss: {current_loss:.4f}"
            )

    avg_epoch_loss = epoch_loss / len(dataloader)
    if writer:
        writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
    return avg_epoch_loss


def validate_one_epoch(
    model, dataloader, criterion, device, image_width, image_height, writer, epoch
):
    model.eval()
    epoch_loss = 0.0

    all_iou_scores = []
    total_samples_for_dev = 0
    sum_edge_dev_metrics = {
        "avg_top_dev": 0.0,
        "avg_bottom_dev": 0.0,
        "avg_left_dev": 0.0,
        "avg_right_dev": 0.0,
        "frac_top_dev_le_1px": 0.0,
        "frac_bottom_dev_le_1px": 0.0,
        "frac_left_dev_le_1px": 0.0,
        "frac_right_dev_le_1px": 0.0,
        "frac_top_dev_le_3px": 0.0,
        "frac_bottom_dev_le_3px": 0.0,
        "frac_left_dev_le_3px": 0.0,
        "frac_right_dev_le_3px": 0.0,
    }
    sum_corner_dev_metrics = {
        "avg_l1_top_left": 0.0,
        "avg_l1_top_right": 0.0,
        "avg_l1_bottom_left": 0.0,
        "avg_l1_bottom_right": 0.0,
        "frac_l1_top_left_le_1px": 0.0,
        "frac_l1_top_right_le_1px": 0.0,
        "frac_l1_bottom_left_le_1px": 0.0,
        "frac_l1_bottom_right_le_1px": 0.0,
        "frac_l1_top_left_le_3px": 0.0,
        "frac_l1_top_right_le_3px": 0.0,
        "frac_l1_bottom_left_le_3px": 0.0,
        "frac_l1_bottom_right_le_3px": 0.0,
    }

    with torch.no_grad():
        for i, (images, targets_dict) in enumerate(dataloader):
            images = images.to(device)

            preds_top, preds_bottom, preds_left, preds_right = model(images)

            loss_top = criterion(preds_top, targets_dict["top"].to(device).float())
            loss_bottom = criterion(
                preds_bottom, targets_dict["bottom"].to(device).float()
            )
            loss_left = criterion(preds_left, targets_dict["left"].to(device).float())
            loss_right = criterion(
                preds_right, targets_dict["right"].to(device).float()
            )
            total_loss = loss_top + loss_bottom + loss_left + loss_right
            epoch_loss += total_loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"    Batch [{i + 1}/{len(dataloader)}], Val Batch Loss: {total_loss.item():.4f}"
                )

            pred_bboxes_batch = utils.predictions_to_bboxes(
                {
                    "top": preds_top,
                    "bottom": preds_bottom,
                    "left": preds_left,
                    "right": preds_right,
                },
                image_width,
                image_height,
                device,
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

    if writer:
        writer.add_scalar("Loss/validation_epoch", avg_epoch_loss, epoch)
        writer.add_scalar("IoU/validation", avg_iou, epoch)

    print("    Edge Deviance:")
    if total_samples_for_dev > 0:
        for key, summed_value in sum_edge_dev_metrics.items():
            metric_val = summed_value / total_samples_for_dev
            if "frac" in key:
                print(f"        {key}: {metric_val:.4f}")
            else:
                print(f"        {key}: {metric_val:.4f} px")
            if writer:
                writer.add_scalar(
                    f"EdgeDeviance/{key.replace('frac_', 'frac_le_')}",
                    metric_val,
                    epoch,
                )

    print("    Corner Deviance:")
    if total_samples_for_dev > 0:
        for key, summed_value in sum_corner_dev_metrics.items():
            metric_val = summed_value / total_samples_for_dev
            if "frac" in key:
                print(f"        {key}: {metric_val:.4f}")
            else:
                print(f"        {key}: {metric_val:.4f} px")
            if writer:
                writer.add_scalar(
                    f"CornerDeviance/{key.replace('frac_', 'frac_le_')}",
                    metric_val,
                    epoch,
                )
    print("    ------------------------")

    metrics = {
        "avg_val_loss": avg_epoch_loss,
        "avg_iou": avg_iou,
    }
    if total_samples_for_dev > 0:
        for key, summed_value in sum_edge_dev_metrics.items():
            metrics[f"edge_{key}"] = summed_value / total_samples_for_dev
        for key, summed_value in sum_corner_dev_metrics.items():
            metrics[f"corner_{key}"] = summed_value / total_samples_for_dev

    return metrics


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

    # --- TensorBoard Writer ---
    log_dir_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(os.path.join(config.LOG_DIR, log_dir_name))
    print(
        f"TensorBoard logs will be saved to: {os.path.join(config.LOG_DIR, log_dir_name)}"
    )

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
    class Config:
        IMAGE_SIZE = config.IMG_SIZE

    model = BoundingBoxAdjustmentModel(
        config=Config(),
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

    # --- Learning Rate Scheduler ---
    # Scheduler 1: Linear Warmup
    scheduler1 = lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # Start LR will be 0.1 * config.LEARNING_RATE
        end_factor=1.0,
        total_iters=config.WARMUP_EPOCHS,
    )
    # Scheduler 2: Cosine Annealing
    cosine_t_max = config.EPOCHS - config.WARMUP_EPOCHS
    if cosine_t_max <= 0:
        # Handle cases where WARMUP_EPOCHS >= EPOCHS, though unlikely with proper config
        print(
            f"Warning: cosine_t_max ({cosine_t_max}) is not positive. CosineAnnealingLR might not behave as expected."
        )
        print("Ensure EPOCHS > WARMUP_EPOCHS in config.")
        # Default to at least 1 epoch for cosine annealing if warmup covers all/most epochs
        cosine_t_max = max(1, config.EPOCHS // 2)

    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_t_max,  # Number of epochs for one cycle of cosine annealing
        eta_min=config.LR_SCHEDULER_ETA_MIN,
    )
    # Sequential Scheduler: Warmup then Cosine Anneal
    # scheduler1 runs for WARMUP_EPOCHS, then scheduler2 runs for the rest.
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[config.WARMUP_EPOCHS],
    )
    print(
        f"Using LR scheduler: Warmup ({config.WARMUP_EPOCHS} epochs) then Cosine Annealing (T_max={cosine_t_max}, eta_min={config.LR_SCHEDULER_ETA_MIN})."
    )

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
            model, train_dataloader, criterion, optimizer, device, writer, epoch
        )
        print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.4f}")
        if writer:  # Log learning rate
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        avg_val_loss = None
        avg_val_iou = None

        if use_validation and val_dataloader:
            print("Starting validation...")
            val_metrics = validate_one_epoch(
                model,
                val_dataloader,
                criterion,
                device,
                config.IMG_SIZE,
                config.IMG_SIZE,
                writer,
                epoch,
            )
            avg_val_loss = val_metrics["avg_val_loss"]
            avg_val_iou = val_metrics["avg_iou"]

            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                checkpoint_path = os.path.join(
                    config.CHECKPOINT_DIR, "best_model_iou.pth"
                )
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

        # Step the scheduler after validation and saving checkpoints
        scheduler.step()

    print("\nTraining finished.")
    if writer:
        writer.close()


if __name__ == "__main__":
    # This allows running train.py directly as a script
    # Before running, ensure you have COCO dataset downloaded and paths in config.py are correct.
    # Also, run `pip install pycocotools` (or the windows equivalent).

    # For a quick test without actual COCO data, you might need to create
    # dummy annotation files and image directories, or modify CocoAdjustmentDataset
    # to work with some placeholder data.
    # The current script will try to load COCO data based on paths.

    main()
