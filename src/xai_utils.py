"""
xai_utils.py — Explainable AI (GradCAM, SHAP) and Fairness Evaluation.

Provides:
  - generate_gradcam()       : GradCAM overlay for CNN feature maps
  - generate_shap_values()   : SHAP DeepExplainer wrapper
  - evaluate_fairness()      : Accuracy measurement across weather conditions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src import config
from src.data_utils import apply_weather_condition

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ─────────────────────────────────────────────────────────────
# 1. GradCAM (Gradient-weighted Class Activation Mapping)
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Hook-free GradCAM using torch.autograd.grad.

    Strategy:
      1. Register a forward hook to capture the intermediate activation tensor.
      2. Remove the hook immediately after the forward pass.
      3. Use torch.autograd.grad() — no tensor-level hooks, no BackwardHookFunctionBackward,
         no inplace ReLU conflicts.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, x, class_idx=None):
        self.model.eval()
        captured = {}

        # Step 1: Capture live activation tensor via forward hook
        def fwd_hook(module, inp, out):
            captured['acts'] = out   # live tensor — NOT detached

        handle = self.target_layer.register_forward_hook(fwd_hook)
        logits = self.model(x)
        handle.remove()              # remove BEFORE backward — prevents hook accumulation

        acts = captured['acts']      # (1, C, H, W), part of the computation graph

        if class_idx is None:
            class_idx = logits.argmax(1).item()

        # Step 2: Compute gradient w.r.t. activation directly (no tensor hooks)
        self.model.zero_grad()
        grads = torch.autograd.grad(
            outputs=logits[0, class_idx],
            inputs=acts,
            retain_graph=False,
            create_graph=False,
        )[0]                         # (1, C, H, W)

        # Step 3: Weighted channel activation map
        weights = grads.mean(dim=[0, 2, 3])         # (C,)
        cam = (acts.detach().squeeze(0) * weights.view(-1, 1, 1)).mean(0).cpu()
        cam = F.relu(cam)
        cam /= cam.max() + 1e-8

        return cam.numpy(), class_idx


def plot_gradcam(model, target_layer, image_tensor, original_img_np, save_path=None):
    """
    Generate and plot GradCAM heatmap over original image.
    """
    cam = GradCAM(model, target_layer)
    heatmap, pred_idx = cam(image_tensor)
    
    import cv2 # Required for heatmap resizing
    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Convert original to uint8 if necessary
    if original_img_np.max() <= 1.0:
        original_img_np = np.uint8(255 * original_img_np)
        
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap_color * 0.4 + original_img_np * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(original_img_np)
    axes[0].set_title(f"Original (Pred: {config.CIFAR10_CLASSES[pred_idx]})")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("GradCAM Heatmap")
    axes[1].axis('off')
    
    axes[2].imshow(superimposed_img)
    axes[2].set_title("Superimposed")
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ─────────────────────────────────────────────────────────────
# 2. SHAP (SHapley Additive exPlanations)
# ─────────────────────────────────────────────────────────────

def run_shap_explainer(model, background_data, test_data, class_names=None, save_path=None):
    """
    Generate SHAP values using DeepExplainer and plot a custom heatmap.
    Requires: pip install shap

    Uses a custom matplotlib plot instead of shap.image_plot() to avoid
    version-specific API breakage in newer SHAP releases.
    """
    if not SHAP_OK:
        print("[Warning] SHAP not installed. Install with: pip install shap")
        return

    print("[SHAP] Calculating explanations (this may take a moment)...")
    model.eval()
    explainer = shap.DeepExplainer(model, background_data)
    # check_additivity=False: DeepLIFT approximation error from AdaptiveAvgPool2d
    shap_values = explainer.shap_values(test_data, check_additivity=False)

    # ── Normalise shap_values to (N, C, H, W) per class ────────────────────
    # Old SHAP: list of n_classes arrays, each (N, C, H, W)
    # New SHAP: single array (N, C, H, W, n_classes) — handle both
    if isinstance(shap_values, list):
        sv_stack = np.stack(shap_values, axis=0)   # (n_cls, N, C, H, W)
    else:
        sv_stack = np.moveaxis(shap_values, -1, 0) # → (n_cls, N, C, H, W)

    # Mean absolute SHAP across all classes → (N, H, W)
    mean_shap_chw = np.mean(np.abs(sv_stack), axis=0)   # (N, C, H, W)
    mean_shap_hw  = mean_shap_chw.mean(axis=1)           # (N, H, W)

    # Denormalise test images for display
    mean_arr = np.array(config.NORMALIZE_MEAN).reshape(1, 1, 1, 3)
    std_arr  = np.array(config.NORMALIZE_STD ).reshape(1, 1, 1, 3)
    test_nhwc = test_data.permute(0, 2, 3, 1).cpu().numpy()
    test_denorm = (test_nhwc * std_arr + mean_arr).clip(0, 1)

    n_show = min(5, test_data.shape[0])
    fig, axes = plt.subplots(n_show, 2, figsize=(6, n_show * 2.8))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        axes[i, 0].imshow(test_denorm[i])
        axes[i, 0].set_title("Original", fontsize=8)
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(mean_shap_hw[i], cmap="hot", interpolation="nearest")
        axes[i, 1].set_title("Mean |SHAP|", fontsize=8)
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.suptitle("SHAP DeepExplainer — Pixel Importance (mean |SHAP| across classes)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("[SHAP] Done.")



# ─────────────────────────────────────────────────────────────
# 3. Fairness Evaluation (Weather / Lighting)
# ─────────────────────────────────────────────────────────────

def evaluate_fairness(model, X_test, y_test, criterion, device):
    """
    Evaluate accuracy across simulated weather conditions.
    """
    conditions = ["normal", "bright", "dark", "foggy", "noisy"]
    results = {}
    
    model.eval()
    print("\n[Fairness Evaluation] Accuracy across weather conditions:")
    
    with torch.no_grad():
        for cond in conditions:
            # Apply condition
            X_cond = apply_weather_condition(X_test, condition=cond)
            
            # Create temporary dataloader
            X_t = torch.from_numpy(X_cond).float()
            y_t = torch.from_numpy(y_test).long()
            ds = torch.utils.data.TensorDataset(X_t, y_t)
            dl = torch.utils.data.DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)
            
            correct = 0
            total = 0
            for images, labels in dl:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            acc = correct / total
            results[cond] = acc
            print(f"  Condition: {cond:<10} | Accuracy: {acc:.4f}")
            
    # Plot results
    plt.figure(figsize=(8, 4))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    bars = plt.bar(list(results.keys()), list(results.values()), color=colors)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.4f}", va='bottom', ha='center', fontsize=9)
    
    plt.title("Fairness Evaluation — Accuracy Drop Across Weather")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(config.PLOTS_DIR / 'p3_fairness_eval.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results
