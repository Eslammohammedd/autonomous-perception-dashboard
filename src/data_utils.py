"""
data_utils.py — Data loading, synthetic generation, augmentation, and DataLoaders
for the Autonomous Vehicle Perception Module.

Provides:
  - load_cifar10()         : load CIFAR-10 via torchvision (auto-download)
  - generate_demo_data()   : structured synthetic 32x32 RGB images
  - add_gaussian_noise()   : noise injection for AutoEncoder training
  - get_dataloaders()      : PyTorch DataLoader factory
  - get_numpy_data()       : flat numpy arrays for scikit-learn
  - apply_weather_condition(): simulate lighting/weather for fairness eval
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
# torchvision is imported lazily inside load_cifar10() to support demo mode without it

from src import config


def _get_train_transform():
    """Lazy-import torchvision.transforms — only called in live (non-demo) mode."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
    ])


def _get_eval_transform():
    """Lazy-import torchvision.transforms — only called in live (non-demo) mode."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
    ])


def load_cifar10(data_dir=None):
    """Load CIFAR-10 via torchvision (auto-downloads if needed)."""
    if data_dir is None:
        data_dir = str(config.RAW_DIR)
    from torchvision import datasets
    train_ds = datasets.CIFAR10(root=data_dir, train=True,  download=True,
                                 transform=_get_train_transform())
    test_ds  = datasets.CIFAR10(root=data_dir, train=False, download=True,
                                 transform=_get_eval_transform())
    print(f"[CIFAR-10] Train={len(train_ds)} | Test={len(test_ds)}")
    return train_ds, test_ds


def load_gtsrb(data_dir=None):
    """Load German Traffic Sign dataset via torchvision.
    Standardizes everything to 32x32 images for consistency.
    """
    if data_dir is None:
        data_dir = str(config.RAW_DIR)
    from torchvision import datasets, transforms
    
    # Custom pipeline that forces standard 32x32 sizing 
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])
    
    # GTSRB does not have explicit train=True/False flags in the same way
    train_ds = datasets.GTSRB(root=data_dir, split="train", download=True, transform=transform)
    test_ds  = datasets.GTSRB(root=data_dir, split="test",  download=True, transform=transform)
    print(f"[GTSRB] Train={len(train_ds)} | Test={len(test_ds)}")
    return train_ds, test_ds


def _make_class_pattern(cls_idx, n, rng):
    """Generate n images with class-specific color + geometric pattern."""
    H, W, C = 32, 32, 3
    base_colors = np.array([
        [135,206,235],[220,50,50],[100,180,100],[200,160,100],
        [140,200,80],[160,100,60],[50,160,50],[180,140,80],
        [30,90,180],[80,80,80],
    ], dtype=np.float32) / 255.0
    color = base_colors[cls_idx]
    imgs = []
    for _ in range(n):
        img = np.zeros((H, W, C), dtype=np.float32)
        for c in range(C):
            img[:,:,c] = color[c] + rng.normal(0, 0.06, (H, W))
        cy, cx = H//2, W//2
        oh, ow = H//3, W//3
        for c in range(C):
            p = color[c]*1.3 + rng.normal(0, 0.04, (oh, ow))
            img[cy-oh//2:cy+oh//2, cx-ow//2:cx+ow//2, c] = p
        img = np.clip(img, 0, 1)
        mean = np.array(config.NORMALIZE_MEAN)
        std  = np.array(config.NORMALIZE_STD)
        img  = (img - mean) / std
        imgs.append(img.transpose(2, 0, 1))
    return np.array(imgs, dtype=np.float32)


def generate_demo_data(n_train=None, n_test=None, seed=None):
    """Generate structured synthetic demo dataset with class-specific patterns."""
    n_train = n_train or config.DEMO_N_TRAIN
    n_test  = n_test  or config.DEMO_N_TEST
    seed    = seed    or config.RANDOM_SEED
    rng = np.random.default_rng(seed)
    n_cls = config.NUM_CLASSES
    ntr, nte = n_train // n_cls, n_test // n_cls
    Xtr, ytr, Xte, yte = [], [], [], []
    for cls in range(n_cls):
        Xtr.append(_make_class_pattern(cls, ntr, rng)); ytr += [cls]*ntr
        Xte.append(_make_class_pattern(cls, nte, rng)); yte += [cls]*nte
    X_train = np.concatenate(Xtr); y_train = np.array(ytr, dtype=np.int64)
    X_test  = np.concatenate(Xte); y_test  = np.array(yte, dtype=np.int64)
    idx = rng.permutation(len(X_train)); X_train, y_train = X_train[idx], y_train[idx]
    idx = rng.permutation(len(X_test));  X_test,  y_test  = X_test[idx],  y_test[idx]
    print(f"[Demo] Train={len(X_train)} | Test={len(X_test)} | Classes={n_cls}")
    return X_train, y_train, X_test, y_test


def add_gaussian_noise(X, std=None, seed=None):
    """Add Gaussian noise for denoising AutoEncoder training."""
    std  = std  or config.AE_NOISE_STD
    seed = seed or config.RANDOM_SEED
    rng  = np.random.default_rng(seed)
    return (X + rng.normal(0, std, X.shape).astype(np.float32))


def _cap_per_class(X, y, max_per_class, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y):
        ci = np.where(y == cls)[0]
        if len(ci) > max_per_class:
            ci = rng.choice(ci, max_per_class, replace=False)
        idx.extend(ci.tolist())
    idx = np.array(idx); rng.shuffle(idx)
    return X[idx], y[idx]


def get_numpy_data(use_demo=None, max_per_class=None, seed=None):
    """Return flat numpy arrays (CHW float32) for classical ML."""
    use_demo      = use_demo      if use_demo      is not None else config.USE_DEMO
    max_per_class = max_per_class or config.MAX_SAMPLES_CLASS
    seed          = seed          or config.RANDOM_SEED
    
    if use_demo:
        X_tr, y_tr, X_te, y_te = generate_demo_data(seed=seed)
    elif getattr(config, 'DATASET_TYPE', 'CIFAR10') == 'GTSRB':
        train_ds, test_ds = load_gtsrb()
        print("[GTSRB] Converting to numpy vectors...")
        
        # Sequentially pull a capped number of samples
        X_tr, y_tr = [], []
        for i in range(min(len(train_ds), max_per_class * 43)):
            img, label = train_ds[i]
            X_tr.append(img.numpy())
            y_tr.append(label)
        X_te, y_te = [], []
        for i in range(min(len(test_ds), max_per_class * 10)):
            img, label = test_ds[i]
            X_te.append(img.numpy())
            y_te.append(label)
            
        X_tr, y_tr = np.array(X_tr, dtype=np.float32), np.array(y_tr, dtype=np.int64)
        X_te, y_te = np.array(X_te, dtype=np.float32), np.array(y_te, dtype=np.int64)
    else:
        train_ds, test_ds = load_cifar10()
        X_tr = train_ds.data.astype(np.float32) / 255.0
        y_tr = np.array(train_ds.targets)
        X_te = test_ds.data.astype(np.float32)  / 255.0
        y_te = np.array(test_ds.targets)
        X_tr = X_tr.transpose(0,3,1,2)
        X_te = X_te.transpose(0,3,1,2)
        
    X_tr, y_tr = _cap_per_class(X_tr, y_tr, max_per_class,     seed)
    X_te, y_te = _cap_per_class(X_te, y_te, max_per_class//5,  seed)
    print(f"[Numpy] X_train={X_tr.shape} | X_test={X_te.shape}")
    return X_tr, y_tr, X_te, y_te


def get_dataloaders(X_train, y_train, X_test, y_test,
                    val_split=0.1, batch_size=None, seed=None):
    """Build train/val/test DataLoaders from numpy arrays."""
    batch_size = batch_size or config.BATCH_SIZE
    seed       = seed       or config.RANDOM_SEED
    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).long()
    Xte = torch.from_numpy(X_test).float()
    yte = torch.from_numpy(y_test).long()
    full = TensorDataset(Xtr, ytr)
    n_val = int(len(full) * val_split)
    gen   = torch.Generator().manual_seed(seed)
    tr_ds, val_ds = random_split(full, [len(full)-n_val, n_val], generator=gen)
    te_ds = TensorDataset(Xte, yte)
    train_l = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_l   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_l  = DataLoader(te_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[DataLoaders] train={len(tr_ds)} val={len(val_ds)} test={len(te_ds)}")
    return train_l, val_l, test_l


def apply_weather_condition(X, condition="normal"):
    """Simulate weather/lighting for fairness evaluation. condition: normal/bright/dark/foggy/noisy."""
    X = X.copy()
    if condition == "normal":   return X
    elif condition == "bright": return X + 0.5
    elif condition == "dark":   return X - 0.5
    elif condition == "foggy":  return X * 0.6 + np.full_like(X, 0.3)
    elif condition == "noisy":
        rng = np.random.default_rng(99)
        return X + rng.normal(0, 0.3, X.shape).astype(np.float32)
    raise ValueError(f"Unknown condition: {condition}")
