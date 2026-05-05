"""
cnn_models.py — Custom CNN, Convolutional AutoEncoder, optimizer comparison,
and feature map visualization for Autonomous Vehicle Perception Module.

Provides:
  - TrafficCNN          : 4-block custom CNN for traffic scene classification
  - ConvAutoEncoder     : Encoder-Decoder for noisy image reconstruction
  - train_epoch()       : one training epoch (classification or AE)
  - eval_epoch()        : one evaluation epoch
  - train_model()       : full training loop with validation curves
  - compare_optimizers(): train with Adam/SGD/RMSProp, compare curves
  - visualize_feature_maps(): hook-based intermediate layer visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import config


# ─────────────────────────────────────────────────────────────
# Custom CNN — 4-block architecture
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU → (optional MaxPool).
    Note: inplace=False on ReLU is intentional — required for GradCAM hooks.
    """
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),   # inplace=False required for backward hooks
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TrafficCNN(nn.Module):
    """
    4-block CNN for CIFAR-10 traffic scene classification.
    Input:  (B, 3, 32, 32)
    Output: (B, num_classes)

    Architecture:
      Block1: 3  → 64  → Pool → 16x16
      Block2: 64 → 128 → Pool → 8x8
      Block3: 128→ 256 → Pool → 4x4
      Block4: 256→ 256 (no pool)
      GAP → FC(512) → Dropout → FC(num_classes)
    """
    def __init__(self, num_classes=None, dropout=None):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES
        dropout     = dropout     or config.DROPOUT

        self.block1 = ConvBlock(3,   64,  pool=True)
        self.block2 = ConvBlock(64,  128, pool=True)
        self.block3 = ConvBlock(128, 256, pool=True)
        self.block4 = ConvBlock(256, 256, pool=False)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(256, 512)
        self.drop    = nn.Dropout(dropout)
        self.fc2     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────
# Convolutional AutoEncoder — for denoising
# ─────────────────────────────────────────────────────────────

class ConvAutoEncoder(nn.Module):
    """
    Symmetric Conv AE for CIFAR-32 denoising.
    Input:  (B, 3, 32, 32) noisy image
    Output: (B, 3, 32, 32) reconstructed image

    Encoder: 3→32→64→128 (with strided conv)
    Bottleneck: latent spatial map
    Decoder: 128→64→32→3 (ConvTranspose2d)
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=2, padding=1), nn.ReLU())   # 16x16
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())   # 8x8
        self.enc3 = nn.Sequential(
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.ReLU())   # 4x4
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128,64, 4, stride=2, padding=1), nn.ReLU())   # 8x8
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU())   # 16x16
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 3,  4, stride=2, padding=1))              # 32x32

    def encode(self, x):
        return self.enc3(self.enc2(self.enc1(x)))

    def decode(self, z):
        return self.dec1(self.dec2(self.dec3(z)))

    def forward(self, x):
        return self.decode(self.encode(x))


# ─────────────────────────────────────────────────────────────
# Training Loops
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, is_ae=False):
    """Run one training epoch. Returns avg loss (and accuracy for classifier)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if is_ae:
            x_noisy, x_clean = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out  = model(x_noisy)
            loss = criterion(out, x_clean)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (batch[0].size(0))

    avg_loss = total_loss / max(len(loader.dataset), 1)
    if is_ae:
        return avg_loss
    return avg_loss, correct / max(total, 1)


def eval_epoch(model, loader, criterion, device, is_ae=False):
    """Run one evaluation epoch. Returns avg loss (and accuracy for classifier)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if is_ae:
                x_noisy, x_clean = batch[0].to(device), batch[1].to(device)
                out  = model(x_noisy)
                loss = criterion(out, x_clean)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                out  = model(x)
                loss = criterion(out, y)
                correct += (out.argmax(1) == y).sum().item()
                total   += y.size(0)
            total_loss += loss.item() * batch[0].size(0)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    if is_ae:
        return avg_loss
    return avg_loss, correct / max(total, 1)


def train_model(model, train_loader, val_loader, optimizer, criterion,
                epochs, device, scheduler=None, is_ae=False, model_name="CNN"):
    """
    Full training loop with validation. Returns history dict.

    Returns:
        history: {'train_loss': [...], 'val_loss': [...],
                  'train_acc': [...], 'val_acc': [...]}
    """
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs+1):
        if is_ae:
            tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, True)
            vl_loss = eval_epoch (model, val_loader,   criterion, device, True)
            print(f"[{model_name}] Ep {epoch:3d}/{epochs} | tr_loss={tr_loss:.4f} | vl_loss={vl_loss:.4f}")
            history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        else:
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            vl_loss, vl_acc = eval_epoch (model, val_loader,   criterion, device)
            print(f"[{model_name}] Ep {epoch:3d}/{epochs} | tr={tr_loss:.4f}/{tr_acc:.4f} | vl={vl_loss:.4f}/{vl_acc:.4f}")
            history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
            history["train_acc" ].append(tr_acc);  history["val_acc" ].append(vl_acc)
        if scheduler:
            if hasattr(scheduler, "step_on_metric"):
                scheduler.step(vl_loss)
            else:
                scheduler.step()
        if vl_loss < best_val:
            best_val   = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return history


def plot_training_curves(histories: dict, metric="loss", save_path=None):
    """
    Plot train/val curves for multiple models.

    Args:
        histories: {model_name: history_dict, ...}
        metric: 'loss' | 'acc'
    """
    fig, axes = plt.subplots(1, len(histories), figsize=(6*len(histories), 4))
    if len(histories) == 1:
        axes = [axes]
    colors = [("#4C72B0","#DD8452"), ("#55A868","#C44E52"), ("#8172B2","#CCB974")]
    for ax, (name, hist), (c1,c2) in zip(axes, histories.items(), colors):
        tr_key = f"train_{metric}"; vl_key = f"val_{metric}"
        if tr_key in hist:
            ax.plot(hist[tr_key], label="Train", color=c1, linewidth=2)
        if vl_key in hist:
            ax.plot(hist[vl_key], label="Val",   color=c2, linewidth=2, linestyle="--")
        ax.set_title(name); ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize()); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────
# Optimizer Comparison
# ─────────────────────────────────────────────────────────────

def _build_optimizer(model, name, lr=None):
    lr = lr or config.LR_DEFAULT
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=config.WEIGHT_DECAY)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr,
                                   weight_decay=config.WEIGHT_DECAY)
    raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(optimizer, name, epochs):
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.3)
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        sch.step_on_metric = True
        return sch
    raise ValueError(f"Unknown scheduler: {name}")


def compare_optimizers(train_loader, val_loader, epochs=None, device=None,
                       optimizers_to_compare=None, save_path=None):
    """
    Train identical CNN architectures with different optimizers.
    Returns dict of histories for plotting.
    """
    epochs   = epochs   or config.EPOCHS_CNN
    device   = device   or config.DEVICE
    opt_list = optimizers_to_compare or config.OPTIMIZERS_TO_COMPARE
    histories = {}

    for opt_name in opt_list:
        print(f"\n{'─'*50}")
        print(f"  Optimizer: {opt_name.upper()}")
        model     = TrafficCNN().to(device)
        optimizer = _build_optimizer(model, opt_name)
        criterion = nn.CrossEntropyLoss()
        hist = train_model(model, train_loader, val_loader, optimizer, criterion,
                           epochs, device, model_name=f"CNN-{opt_name}")
        histories[f"CNN-{opt_name.upper()}"] = hist

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(opt_list)))
    for (name, hist), color in zip(histories.items(), colors):
        ax1.plot(hist["val_loss"], label=name, color=color, linewidth=2)
        if hist.get("val_acc"):
            ax2.plot(hist["val_acc"], label=name, color=color, linewidth=2)
    ax1.set_title("Val Loss — Optimizer Comparison"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_title("Val Accuracy — Optimizer Comparison"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return histories


def compare_schedulers(train_loader, val_loader, epochs=None, device=None,
                        schedulers_to_compare=None, save_path=None):
    """Train identical CNNs with Adam + different LR schedulers."""
    epochs   = epochs   or config.EPOCHS_CNN
    device   = device   or config.DEVICE
    sch_list = schedulers_to_compare or config.SCHEDULERS_TO_COMPARE
    histories = {}

    for sch_name in sch_list:
        print(f"\n  Scheduler: {sch_name}")
        model     = TrafficCNN().to(device)
        optimizer = _build_optimizer(model, "adam")
        scheduler = _build_scheduler(optimizer, sch_name, epochs)
        criterion = nn.CrossEntropyLoss()
        hist = train_model(model, train_loader, val_loader, optimizer, criterion,
                           epochs, device, scheduler=scheduler,
                           model_name=f"CNN-{sch_name}")
        histories[f"Sched-{sch_name}"] = hist

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, hist in histories.items():
        if hist.get("val_acc"):
            ax.plot(hist["val_acc"], label=name, linewidth=2)
    ax.set_title("Val Accuracy — LR Scheduler Comparison")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return histories


# ─────────────────────────────────────────────────────────────
# Feature Map Visualization
# ─────────────────────────────────────────────────────────────

def visualize_feature_maps(model, image_tensor, layer_name="block1",
                            n_maps=16, save_path=None):
    """
    Visualize intermediate CNN feature maps using forward hooks.

    Args:
        model       : TrafficCNN (fitted)
        image_tensor: (1, C, H, W) tensor
        layer_name  : attribute name of the layer to hook (e.g. 'block1')
        n_maps      : number of feature maps to display
    """
    activations = {}
    def hook_fn(module, inp, out):
        activations["output"] = out.detach().cpu()

    layer = getattr(model, layer_name, None)
    if layer is None:
        print(f"[Warning] Layer '{layer_name}' not found in model.")
        return
    handle = layer.register_forward_hook(hook_fn)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(image_tensor.to(device))
    handle.remove()

    fmaps = activations["output"][0]   # (C, H, W)
    n_show = min(n_maps, fmaps.shape[0])
    ncols  = 8
    nrows  = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*1.5, nrows*1.5))
    axes = axes.flatten()
    for i in range(n_show):
        fmap = fmaps[i].numpy()
        axes[i].imshow(fmap, cmap="viridis", aspect="auto")
        axes[i].set_title(f"Ch {i}", fontsize=7)
        axes[i].axis("off")
    for i in range(n_show, len(axes)):
        axes[i].axis("off")
    plt.suptitle(f"Feature Maps — {layer_name}", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
