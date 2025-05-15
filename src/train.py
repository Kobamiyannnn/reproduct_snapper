# src/train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np

# Assuming config.py, models.py, dataset.py, utils.py are in the same src directory
# or PYTHONPATH is correctly set.
from . import config
from .models import BoundingBoxAdjustmentModel
from .dataset import CocoAdjustmentDataset
# from .utils import calculate_iou # Example, if you add evaluation metrics in utils


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


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    # TODO: Implement proper evaluation metrics (e.g., IoU, Edge Deviance)
    # all_predictions_for_metrics = []
    # all_targets_for_metrics = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            target_top = targets["top"].to(device)
            target_bottom = targets["bottom"].to(device)
            target_left = targets["left"].to(device)
            target_right = targets["right"].to(device)

            predictions = model(images)

            loss_t = criterion(predictions["top"], target_top)
            loss_b = criterion(predictions["bottom"], target_bottom)
            loss_l = criterion(predictions["left"], target_left)
            loss_r = criterion(predictions["right"], target_right)
            total_loss = loss_t + loss_b + loss_l + loss_r
            epoch_loss += total_loss.item()

            if (i + 1) % 50 == 0:  # Log every 50 batches for validation
                print(
                    f"    Batch [{i + 1}/{len(dataloader)}], Val Batch Loss: {total_loss.item():.4f}"
                )

            # For metric calculation:
            # 1. Convert predictions (logits) to probabilities (e.g., torch.sigmoid)
            # 2. Convert probabilities to coordinates (e.g., argmax or expectation over the probability distribution)
            #    - pred_top_coords = torch.argmax(torch.sigmoid(predictions["top"]), dim=1)
            #    - Similar for bottom, left, right
            # 3. Store these predicted coordinates and the original GT coordinates (derived from target vectors or passed separately)
            #    The target vectors are one-hot, so GT coords are torch.argmax(target_*, dim=1)
            # all_predictions_for_metrics.append(predicted_bboxes_batch)
            # all_targets_for_metrics.append(gt_bboxes_batch)

    avg_epoch_loss = epoch_loss / len(dataloader)

    # After the loop, calculate metrics using all_predictions_for_metrics and all_targets_for_metrics
    # avg_iou = calculate_iou(all_predictions_for_metrics, all_targets_for_metrics) # Placeholder
    # print(f"    Validation Avg IoU: {avg_iou:.4f}") # Placeholder

    return avg_epoch_loss  # , avg_iou (or other metrics)


def main():
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
    best_val_metric = float(
        "inf"
    )  # Lower is better for loss, or float('-inf') if higher is better for IoU

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")

        avg_train_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device
        )
        print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.4f}")

        current_metric_for_saving = (
            avg_train_loss  # Default to train loss if no validation
        )

        if use_validation and val_dataloader:
            avg_val_loss = validate_one_epoch(model, val_dataloader, criterion, device)
            # avg_val_loss, avg_val_iou = validate_one_epoch(model, val_dataloader, criterion, device) # If returning metrics
            print(f"Epoch {epoch} - Average Validation Loss: {avg_val_loss:.4f}")
            # print(f"Epoch {epoch} - Average Validation IoU: {avg_val_iou:.4f}") # If using IoU
            current_metric_for_saving = (
                avg_val_loss  # Use validation loss for choosing best model
            )

        # --- Save Model Checkpoint ---
        # Save best model based on validation metric (e.g., loss or IoU)
        if use_validation and current_metric_for_saving < best_val_metric:
            best_val_metric = current_metric_for_saving
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": current_metric_for_saving,
                },
                checkpoint_path,
            )
            print(
                f"Saved best model checkpoint (Val Metric: {best_val_metric:.4f}) to {checkpoint_path}"
            )

        # Save model periodically
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, f"model_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,  # Save training loss with periodic checkpoints
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
