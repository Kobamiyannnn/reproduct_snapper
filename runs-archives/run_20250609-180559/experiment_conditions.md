# Experiment Conditions

## Data Settings
- COCO_ANNOTATIONS_PATH_TRAIN: `data/coco/annotations/instances_train2017.json`
- COCO_IMG_DIR_TRAIN: `data/coco/train2017/`
- COCO_ANNOTATIONS_PATH_VAL: `data/coco/annotations/instances_val2017.json`
- COCO_IMG_DIR_VAL: `data/coco/val2017/`

## Model Input Settings
- IMG_SIZE: 224
- MEAN: [0.485, 0.456, 0.406]
- STD: [0.229, 0.224, 0.225]

## Model Architecture Settings (from config)
- BASE_MODEL_NAME: `resnet50`
- NUM_FEATURES (potentially legacy): 2048
- DECODER_CHANNELS (potentially legacy): [512, 128]

## Actual Model Architecture (from models.py implementation)
- Backbone: ResNet-50 (using layer3 and layer4 outputs)
- Layer3 Feature Dim: 1024
- Layer4 Feature Dim: 2048
- Decoder Type: nn.Linear per edge, per scale
- Multi-Scale Integration: Averaging predictions from layer3 and layer4 decoders

## Training Hyperparameters
- LEARNING_RATE: 0.0002
- EPOCHS: 50
- BATCH_SIZE: 128
- OPTIMIZER_TYPE: `Adam`
- LOSS_FN_TYPE: `BCEWithLogitsLoss`
- WARMUP_EPOCHS: 5
- LR_SCHEDULER_ETA_MIN: 1e-06

## Jittering and Cropping Settings
- CENTER_JITTER_RATIO: 0.1
- SCALE_JITTER_RATIO: 0.1
- BUFFER_RATIO: 1.1

## Environment Settings
- DEVICE: `cuda`
- NUM_WORKERS: 4
- RANDOM_SEED: 42
