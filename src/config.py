# src/config.py

# --- Data Settings ---
COCO_ANNOTATIONS_PATH_TRAIN = "data/coco/annotations/instances_train2017.json"
COCO_IMG_DIR_TRAIN = "data/coco/train2017/"
COCO_ANNOTATIONS_PATH_VAL = "data/coco/annotations/instances_val2017.json"
COCO_IMG_DIR_VAL = "data/coco/val2017/"

# --- Model Input Settings ---
CROP_IMG_HEIGHT = 224  # Height of the image cropped and resized for model input
CROP_IMG_WIDTH = 224  # Width of the image cropped and resized for model input
MODEL_FEATURE_MAP_DOWNSAMPLE_RATIO = (
    32  # ResNet50 typically has a downsample ratio of 32
)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
EPOCHS = 5  # Default: 50
BATCH_SIZE = 64  # Adjust based on your GPU memory
OPTIMIZER_TYPE = "Adam"  # "Adam", "SGD", etc.
LOSS_FN_TYPE = "BCEWithLogitsLoss"

# --- Jittering and Cropping Settings ---
CENTER_JITTER_RATIO = 0.1  # Max percentage of bbox dimension to shift the center
SCALE_JITTER_RATIO = (
    0.1  # For rescaling, e.g., 0.1 means scale factor between 0.9 and 1.1
)
BUFFER_RATIO = (
    1.1  # Ratio to expand the rough bbox for cropping (can be a range for augmentation)
)

# --- Checkpoint Settings ---
CHECKPOINT_DIR = "checkpoints/"
SAVE_EVERY_N_EPOCHS = 5  # Save model every N epochs

# --- Evaluation Settings ---
# EVAL_IOU_THRESHOLD = 0.5 # Example for some evaluation metrics

# --- Environment Settings ---
DEVICE = "cuda"  # "cuda" if GPU is available, else "cpu"
NUM_WORKERS = 4  # For DataLoader, adjust based on your CPU cores
RANDOM_SEED = 42
