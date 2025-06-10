# src/resume_train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim.lr_scheduler as lr_scheduler
import argparse

# Assuming config.py, models.py, dataset.py, utils.py are in the same src directory
import config
from models import BoundingBoxAdjustmentModel
from dataset import CocoAdjustmentDataset
from train import seed_everything, train_one_epoch, validate_one_epoch


def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    """
    チェックポイントファイルから状態を復元する
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # モデルの状態を復元
    model.load_state_dict(checkpoint["model_state_dict"])

    # オプティマイザーの状態を復元
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # エポック番号を取得
    start_epoch = checkpoint["epoch"]

    # スケジューラーの状態を復元（存在する場合）
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # その他の情報を取得
    best_val_iou = checkpoint.get("iou", float("-inf"))

    print("Checkpoint loaded successfully!")
    print(f"  - Epoch: {start_epoch}")
    print(f"  - Best validation IoU: {best_val_iou:.4f}")

    return start_epoch, best_val_iou


def find_existing_log_dir(checkpoint_path):
    """
    チェックポイントファイルから対応するTensorBoardログディレクトリを推定する
    """
    # チェックポイントファイルのディレクトリから run_* 部分を抽出
    checkpoint_dir = os.path.dirname(checkpoint_path)
    run_dir_name = os.path.basename(checkpoint_dir)

    if run_dir_name.startswith("run_"):
        log_dir_path = os.path.join(config.LOG_DIR, run_dir_name)
        if os.path.exists(log_dir_path):
            return log_dir_path, run_dir_name

    return None, None


def save_checkpoint_with_scheduler(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    train_loss,
    val_loss=None,
    val_iou=None,
):
    """
    スケジューラーの状態も含めてチェックポイントを保存する
    """
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "train_loss": train_loss,
        "val_loss": val_loss if val_loss is not None else -1,
        "iou": val_iou if val_iou is not None else -1,
    }
    torch.save(checkpoint_data, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Existing TensorBoard log directory to continue (optional)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return

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

    print("Starting resume training process...")
    seed_everything(config.RANDOM_SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- TensorBoard Writer ---
    if args.log_dir:
        log_full_path = args.log_dir
        log_dir_name = os.path.basename(log_full_path)
        print(f"Using specified TensorBoard log directory: {log_full_path}")
    else:
        # チェックポイントから既存のログディレクトリを推定
        existing_log_dir, log_dir_name = find_existing_log_dir(args.checkpoint)
        if existing_log_dir:
            log_full_path = existing_log_dir
            print(f"Found existing TensorBoard log directory: {log_full_path}")
        else:
            # 新しいログディレクトリを作成
            log_dir_name = f"resume_{time.strftime('%Y%m%d-%H%M%S')}"
            log_full_path = os.path.join(config.LOG_DIR, log_dir_name)
            if not os.path.exists(log_full_path):
                os.makedirs(log_full_path)
            print(f"Created new TensorBoard log directory: {log_full_path}")

    writer = SummaryWriter(log_full_path)

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
            for_training=False,
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

    # --- Model ---
    class ConfigModelInit:
        IMAGE_SIZE = config.IMG_SIZE

    model = BoundingBoxAdjustmentModel(config=ConfigModelInit()).to(device)
    print("Model initialized.")

    # --- Loss, Optimizer, Scheduler ---
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
    scheduler1 = lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.WARMUP_EPOCHS,
    )
    cosine_t_max = config.EPOCHS - config.WARMUP_EPOCHS
    if cosine_t_max <= 0:
        print(
            f"Warning: cosine_t_max ({cosine_t_max}) is not positive. CosineAnnealingLR might not behave as expected."
        )
        print("Ensure EPOCHS > WARMUP_EPOCHS in config.")
        cosine_t_max = max(1, config.EPOCHS // 2)

    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_t_max,
        eta_min=config.LR_SCHEDULER_ETA_MIN,
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[config.WARMUP_EPOCHS],
    )
    print(
        f"Using LR scheduler: Warmup ({config.WARMUP_EPOCHS} epochs) then Cosine Annealing (T_max={cosine_t_max}, eta_min={config.LR_SCHEDULER_ETA_MIN})."
    )

    # --- Load Checkpoint ---
    start_epoch, best_val_iou = load_checkpoint(
        args.checkpoint, model, optimizer, scheduler
    )

    # --- Checkpoint Directory ---
    # 既存のチェックポイントディレクトリまたは新しく作成
    if log_dir_name:
        checkpoint_dir_full_path = os.path.join(config.CHECKPOINT_DIR, log_dir_name)
    else:
        checkpoint_dir_full_path = os.path.dirname(args.checkpoint)

    if not os.path.exists(checkpoint_dir_full_path):
        os.makedirs(checkpoint_dir_full_path)
        print(f"Created checkpoint directory: {checkpoint_dir_full_path}")
    else:
        print(f"Using existing checkpoint directory: {checkpoint_dir_full_path}")

    # --- Resume情報を記録 ---
    resume_info_path = os.path.join(log_full_path, "resume_info.md")
    with open(resume_info_path, "a") as f:  # append mode
        f.write(f"\n## Resume Information - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Resumed from checkpoint: `{args.checkpoint}`\n")
        f.write(f"- Starting epoch: {start_epoch + 1}\n")
        f.write(f"- Best validation IoU before resume: {best_val_iou:.4f}\n")
        f.write(f"- Target epochs: {config.EPOCHS}\n\n")
    print(f"Resume information saved to: {resume_info_path}")

    # --- Training Loop ---
    print(f"Resuming training from epoch {start_epoch + 1} to {config.EPOCHS}...")

    for epoch in range(start_epoch + 1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} (Resume) ---")

        avg_train_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, writer, epoch
        )
        print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.4f}")
        if writer:
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
            avg_val_iou = val_metrics["mIoU"]

            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                checkpoint_path = os.path.join(
                    checkpoint_dir_full_path, "best_model_iou.pth"
                )
                save_checkpoint_with_scheduler(
                    checkpoint_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    avg_train_loss,
                    avg_val_loss,
                    avg_val_iou,
                )
                print(
                    f"Saved best model checkpoint (Val IoU: {best_val_iou:.4f}) to {checkpoint_path}"
                )

        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir_full_path, f"model_epoch_{epoch}.pth"
            )
            save_checkpoint_with_scheduler(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                avg_train_loss,
                avg_val_loss,
                avg_val_iou,
            )
            print(f"Saved periodic model checkpoint to {checkpoint_path}")

        scheduler.step()

    print("\nResume training finished.")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
