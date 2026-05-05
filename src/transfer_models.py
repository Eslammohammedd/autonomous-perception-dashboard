"""
transfer_models.py — MobileNetV2 & VGG16 fine-tuning for AV Perception Module.

Before/After transfer learning comparison showing accuracy improvement.

Provides:
  - build_mobilenet_v2()    : load pretrained MobileNetV2, replace head
  - build_vgg16()           : load pretrained VGG16, replace classifier head
  - fine_tune_model()       : two-stage fine-tuning (head only → full)
  - evaluate_transfer_model(): compute accuracy + F1 on test set
  - compare_transfer_models(): before/after + side-by-side bar chart
  - plot_lr_scheduler_curves(): learning rate curves visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# torchvision.models is imported lazily inside builder functions (Phase 2 only)
from sklearn.metrics import accuracy_score, f1_score

from src import config
from src.cnn_models import train_epoch, eval_epoch, plot_training_curves


# ─────────────────────────────────────────────────────────────
# Model Builders
# ─────────────────────────────────────────────────────────────

def build_mobilenet_v2(num_classes=None, freeze_backbone=True):
    """
    Load pretrained MobileNetV2, replace the classifier head.
    Requires torchvision: pip install torchvision

    Args:
        num_classes    : output classes (default config.NUM_CLASSES)
        freeze_backbone: freeze all layers except new head initially
    Returns:
        model (nn.Module)
    """
    try:
        from torchvision import models
    except ImportError:
        raise ImportError("torchvision required: pip install torchvision")
    num_classes = num_classes or config.NUM_CLASSES
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(config.DROPOUT),
        nn.Linear(in_features, num_classes),
    )
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad_(False)
    return model


def build_vgg16(num_classes=None, freeze_backbone=True):
    """
    Load pretrained VGG16, replace the final classification layer.
    Requires torchvision: pip install torchvision

    Args:
        num_classes    : output classes
        freeze_backbone: freeze features layers initially
    Returns:
        model (nn.Module)
    """
    try:
        from torchvision import models
    except ImportError:
        raise ImportError("torchvision required: pip install torchvision")
    num_classes = num_classes or config.NUM_CLASSES
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad_(False)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _unfreeze_all(model):
    """Unfreeze all parameters for full fine-tuning stage."""
    for param in model.parameters():
        param.requires_grad_(True)


# ─────────────────────────────────────────────────────────────
# Training — Two-Stage Fine-Tuning
# ─────────────────────────────────────────────────────────────

def fine_tune_model(model, train_loader, val_loader,
                    epochs_stage1=None, epochs_stage2=None,
                    lr_stage1=1e-3, lr_stage2=1e-5,
                    device=None, model_name="TransferModel"):
    """
    Two-stage fine-tuning:
      Stage 1: Train head only (backbone frozen)    — higher LR
      Stage 2: Unfreeze all, train end-to-end       — very low LR

    Returns:
        model, combined_history dict
    """
    epochs_stage1 = epochs_stage1 or max(2, config.EPOCHS_TL // 2)
    epochs_stage2 = epochs_stage2 or max(1, config.EPOCHS_TL - epochs_stage1)
    device        = device        or config.DEVICE

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Stage 1: Head only ──────────────────────────────────
    print(f"\n[{model_name}] Stage 1 — Head only (lr={lr_stage1})")
    optim1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage1, weight_decay=config.WEIGHT_DECAY
    )
    for ep in range(1, epochs_stage1 + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim1, criterion, device)
        vl_loss, vl_acc = eval_epoch (model, val_loader,   criterion, device)
        print(f"  Stage1 Ep {ep}/{epochs_stage1} | tr={tr_acc:.4f} vl={vl_acc:.4f}")
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_acc" ].append(tr_acc);  history["val_acc" ].append(vl_acc)

    # ── Stage 2: Full fine-tune ─────────────────────────────
    _unfreeze_all(model)
    print(f"\n[{model_name}] Stage 2 — Full fine-tune (lr={lr_stage2})")
    optim2 = torch.optim.Adam(model.parameters(), lr=lr_stage2,
                               weight_decay=config.WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=epochs_stage2)
    for ep in range(1, epochs_stage2 + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim2, criterion, device)
        vl_loss, vl_acc = eval_epoch (model, val_loader,   criterion, device)
        sched2.step()
        print(f"  Stage2 Ep {ep}/{epochs_stage2} | tr={tr_acc:.4f} vl={vl_acc:.4f}")
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_acc" ].append(tr_acc);  history["val_acc" ].append(vl_acc)

    return model, history


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_transfer_model(model, test_loader, device=None, model_name="Model"):
    """
    Evaluate a fine-tuned transfer model on test set.

    Returns:
        dict with accuracy, f1_macro, f1_weighted
    """
    device = device or config.DEVICE
    model.eval().to(device)
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            pred   = logits.argmax(1).cpu().numpy()
            all_pred.extend(pred.tolist())
            all_true.extend(y.numpy().tolist())

    acc    = accuracy_score(all_true, all_pred)
    f1_mac = f1_score(all_true, all_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    result = {
        "model":       model_name,
        "accuracy":    round(acc,    4),
        "f1_macro":    round(f1_mac, 4),
        "f1_weighted": round(f1_wt,  4),
    }
    print(f"\n[{model_name}] Acc={acc:.4f} | F1(macro)={f1_mac:.4f} | F1(weighted)={f1_wt:.4f}")
    return result


# ─────────────────────────────────────────────────────────────
# Before / After Transfer Comparison
# ─────────────────────────────────────────────────────────────

def compare_transfer_models(results_before: dict, results_after: dict,
                             save_path=None):
    """
    Bar chart showing before vs after transfer learning accuracy for each model.

    Args:
        results_before: {model_name: accuracy, ...} — random-init baseline
        results_after : {model_name: accuracy, ...} — after fine-tuning
    """
    models_list = list(results_after.keys())
    before = [results_before.get(m, 0) for m in models_list]
    after  = [results_after[m]          for m in models_list]

    x    = np.arange(len(models_list))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(models_list)*2.5), 5))
    b1 = ax.bar(x - w/2, before, w, label="Before Transfer (Random Init)",
                color="#DD8452", edgecolor="white")
    b2 = ax.bar(x + w/2, after,  w, label="After Fine-Tuning",
                color="#4C72B0", edgecolor="white")
    ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models_list)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.set_title("Transfer Learning — Before vs After Fine-Tuning")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def get_random_init_accuracy(model_builder_fn, test_loader, device=None):
    """
    Evaluate a randomly-initialized model (no pretraining) as the 'before' baseline.
    """
    device = device or config.DEVICE
    model  = model_builder_fn(freeze_backbone=False)
    # Override pretrained weights with random init
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return evaluate_transfer_model(model, test_loader, device, "RandomInit")
