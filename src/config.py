"""
config.py — Central configuration for Autonomous Vehicle Perception Module.

Traffic Scene Classification using CIFAR-10 as a lightweight proxy.
Modify USE_DEMO and FAST_MODE to control data source and training speed.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
DEMO_DIR    = DATA_DIR / "demo"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR  = RESULTS_DIR / "models"
PLOTS_DIR   = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

for d in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR, DEMO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
# USE_DEMO = True  → generate synthetic images (no download needed)
# USE_DEMO = False → use real data via torchvision (auto-downloads)
USE_DEMO = False

# Real dataset selector: "CIFAR10" or "GTSRB"
DATASET_TYPE = "GTSRB" 


# CIFAR-10 classes remapped as traffic-scene objects
# Disclaimer: CIFAR-10 is used as a lightweight proxy for perception tasks.
# A real AV system would use KITTI, BDD100K, or nuScenes.
CIFAR10_CLASSES = [
    "airplane",    # → aerial object (drone/aircraft over road)
    "automobile",  # → car
    "bird",        # → small flying object
    "cat",         # → small animal on road
    "deer",        # → large animal on road
    "dog",         # → medium animal on road
    "frog",        # → small obstacle
    "horse",       # → large animal / slow vehicle
    "ship",        # → large waterway vehicle
    "truck",       # → heavy vehicle / commercial truck
]

GTSRB_CLASSES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing", "No passing vehicles over 3.5 tons", "Priority at next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles > 3.5 tonnes prohibited", "No entry",
    "General caution", "Dangerous curve left", "Dangerous curve right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End speed + passing limits", "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End no passing vehicles > 3.5 tons"
]

def get_current_classes():
    # Fix NameError: use global scope instead of self-referential getattr
    if globals().get("DATASET_TYPE", "CIFAR10") == "GTSRB":
        return GTSRB_CLASSES
    return CIFAR10_CLASSES


# Human-readable scene labels for display
SCENE_LABELS = [
    "Aerial Object", "Car", "Small Flying", "Small Animal",
    "Large Animal", "Medium Animal", "Small Obstacle",
    "Large Animal/Slow", "Watercraft", "Heavy Truck"
]

NUM_CLASSES = 43 if globals().get("DATASET_TYPE", "CIFAR10") == "GTSRB" else 10

# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────
IMG_SIZE       = (32, 32)      # CIFAR-10 native resolution
IMG_CHANNELS   = 3             # RGB
# ImageNet normalization (used for transfer learning)
NORMALIZE_MEAN = [0.4914, 0.4822, 0.4465]   # CIFAR-10 channel means
NORMALIZE_STD  = [0.2023, 0.1994, 0.2010]   # CIFAR-10 channel stds

# ─────────────────────────────────────────────
# Phase 1 — Classical ML
# ─────────────────────────────────────────────
HOG_ORIENTATIONS  = 9
HOG_PIXELS_CELL   = (8, 8)
HOG_CELLS_BLOCK   = (2, 2)
COLOR_HIST_BINS   = 32        # bins per channel
LBP_RADIUS        = 2
LBP_N_POINTS      = 8 * LBP_RADIUS
PCA_VARIANCE      = 0.95      # retain 95% variance by default
PCA_VARIANCE_SWEEP = [0.99, 0.95, 0.90, 0.80]  # for accuracy vs compression curve
MAX_SAMPLES_CLASS = 500       # cap per class for classical ML speed

# KNN
KNN_K_VALUES = [3, 5, 7, 9, 11]

# Grid search param grids
RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}
GBM_PARAM_GRID = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5],
}

# ─────────────────────────────────────────────
# Phase 2 — Deep Learning
# ─────────────────────────────────────────────
BATCH_SIZE   = 128
EPOCHS_CNN   = 30
EPOCHS_AE    = 20
EPOCHS_TL    = 15        # Transfer learning fine-tune
LR_DEFAULT   = 1e-3
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-4

# Demo / fast mode (fewer epochs for quick testing)
FAST_MODE = True
if FAST_MODE:
    EPOCHS_CNN = 5
    EPOCHS_AE  = 5
    EPOCHS_TL  = 3

# Optimizers to compare
OPTIMIZERS_TO_COMPARE = ["adam", "sgd", "rmsprop"]

# LR Schedulers to compare
SCHEDULERS_TO_COMPARE = ["step", "cosine", "plateau"]

# AutoEncoder noise
AE_NOISE_STD = 0.15        # Gaussian noise std for denoising AE

# Transfer Learning models
TL_MODELS = ["mobilenet_v2", "vgg16"]
TL_LR     = 1e-4           # Lower LR for fine-tuning

# ─────────────────────────────────────────────
# Demo data (synthetic)
# ─────────────────────────────────────────────
DEMO_N_TRAIN = 2000        # synthetic training samples
DEMO_N_TEST  = 400         # synthetic test samples

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Phase 4 — YOLOv8 Real-time Perception
# ─────────────────────────────────────────────
# Path to the newly trained model weights
# The 'runs' folder is located in the workspace root (one level above ROOT_DIR)
MODEL_PATH = ROOT_DIR.parent / "runs" / "detect" / "AV_Perception" / "AV_Starter_Model-3" / "weights" / "best.pt"

# Fallback if the above path doesn't exist yet (check locally then pretrained)
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT_DIR / "runs" / "detect" / "AV_Perception" / "AV_Starter_Model-3" / "weights" / "best.pt"
    if not MODEL_PATH.exists():
        MODEL_PATH = "yolov8n.pt"

# Confidence and IOU thresholds
CONF_THRESHOLD = 0.40
IOU_THRESHOLD  = 0.45

# Class filtering (optional: only detect these indices)
# COCO indices: 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11] 
