# src/config.py

# --- Data Settings ---
COCO_ANNOTATIONS_PATH_TRAIN = "data/coco/annotations/instances_train2017.json"
COCO_IMG_DIR_TRAIN = "data/coco/train2017/"
COCO_ANNOTATIONS_PATH_VAL = "data/coco/annotations/instances_val2017.json"
COCO_IMG_DIR_VAL = "data/coco/val2017/"

# --- Model Input Settings ---
IMG_SIZE = (
    224  # Size of the image cropped and resized for model input (height and width)
)
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean for normalization
STD = [0.229, 0.224, 0.225]  # ImageNet std for normalization

# --- Model Architecture Settings ---
BASE_MODEL_NAME = "resnet50"  # Backbone model name
NUM_FEATURES = (
    2048  # Number of features from backbone output (e.g., ResNet50's avgpool output)
)
DECODER_CHANNELS = [
    512,
    128,
]  # List of hidden channels for the 1D Decoders, e.g. [512, 128] means two hidden Conv1D layers

# --- Training Hyperparameters ---
LEARNING_RATE = 2e-4
EPOCHS = 50  # Default: 50
BATCH_SIZE = 128  # Increased batch size
OPTIMIZER_TYPE = "Adam"  # "Adam", "SGD", etc.
LOSS_FN_TYPE = "BCEWithLogitsLoss"
WARMUP_EPOCHS = 5  # Epochs for learning rate warmup
LR_SCHEDULER_ETA_MIN = 1e-6  # Minimum learning rate for CosineAnnealingLR

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
IOU_THRESHOLD = 0.9  # IoU threshold for the 'IoU > X' metric
DEVIANCE_THRESHOLDS = (1.0, 3.0)  # Pixel deviation thresholds for Edge/Corner accuracy

# --- Environment Settings ---
DEVICE = "cuda"  # "cuda" if GPU is available, else "cpu"
NUM_WORKERS = 4  # For DataLoader, adjust based on your CPU cores
RANDOM_SEED = 42

# TensorBoard Log Directory
LOG_DIR = "runs"  # TensorBoardのログが保存されるベースディレクトリ
