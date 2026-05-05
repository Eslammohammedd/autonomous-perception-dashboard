"""
feature_extraction.py — HOG, color histogram, LBP, PCA, t-SNE for AV Perception Module.

Provides:
  - extract_hog()            : Histogram of Oriented Gradients per image
  - extract_color_histogram(): Per-channel color histograms
  - extract_lbp()            : Local Binary Patterns
  - extract_all_features()   : Concatenate all features into one vector
  - apply_pca()              : Fit/transform PCA with variance sweep
  - plot_pca_variance()      : Accuracy vs n_components visualization
  - plot_tsne()              : t-SNE 2D embedding visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    from skimage.feature import hog, local_binary_pattern
    from skimage.color import rgb2gray
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

from src import config


# ─────────────────────────────────────────────────────────────
# Feature Extraction Primitives
# ─────────────────────────────────────────────────────────────

def _chw_to_hwc(X):
    """Convert (N,C,H,W) → (N,H,W,C) for scikit-image."""
    if X.ndim == 4 and X.shape[1] in (1, 3):
        return X.transpose(0, 2, 3, 1)
    return X


def extract_hog(X, orientations=None, pixels_per_cell=None, cells_per_block=None):
    """
    Extract HOG features from each image.

    Args:
        X: (N, C, H, W) float32 images
    Returns:
        features: (N, hog_dim) float64
    """
    if not SKIMAGE_OK:
        raise ImportError("scikit-image required: pip install scikit-image")
    orientations    = orientations    or config.HOG_ORIENTATIONS
    pixels_per_cell = pixels_per_cell or config.HOG_PIXELS_CELL
    cells_per_block = cells_per_block or config.HOG_CELLS_BLOCK

    X_hwc = _chw_to_hwc(X)
    feats = []
    for img in X_hwc:
        # Denormalize to [0,1] for HOG
        img_denorm = img.clip(-3, 3)
        img_denorm = (img_denorm - img_denorm.min()) / (img_denorm.max() - img_denorm.min() + 1e-8)
        if img_denorm.shape[2] == 3:
            gray = rgb2gray(img_denorm)
        else:
            gray = img_denorm[:, :, 0]
        h = hog(gray, orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                feature_vector=True)
        feats.append(h)
    return np.array(feats, dtype=np.float64)


def extract_color_histogram(X, bins=None):
    """
    Extract per-channel color histograms and concatenate.

    Args:
        X: (N, C, H, W) float32
    Returns:
        (N, C*bins) float64
    """
    bins = bins or config.COLOR_HIST_BINS
    X_hwc = _chw_to_hwc(X)
    feats = []
    for img in X_hwc:
        img_denorm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        hist_channels = []
        n_c = img_denorm.shape[2] if img_denorm.ndim == 3 else 1
        for c in range(n_c):
            ch = img_denorm[:, :, c] if img_denorm.ndim == 3 else img_denorm
            h, _ = np.histogram(ch, bins=bins, range=(0, 1))
            hist_channels.append(h.astype(np.float64) / (h.sum() + 1e-8))
        feats.append(np.concatenate(hist_channels))
    return np.array(feats, dtype=np.float64)


def extract_lbp(X, radius=None, n_points=None, n_bins=64):
    """
    Extract Local Binary Pattern histogram per image.

    Args:
        X: (N, C, H, W) float32
    Returns:
        (N, n_bins) float64
    """
    if not SKIMAGE_OK:
        raise ImportError("scikit-image required: pip install scikit-image")
    radius   = radius   or config.LBP_RADIUS
    n_points = n_points or config.LBP_N_POINTS
    X_hwc = _chw_to_hwc(X)
    feats = []
    for img in X_hwc:
        img_denorm = img.clip(-3, 3)
        img_denorm = (img_denorm - img_denorm.min()) / (img_denorm.max() - img_denorm.min() + 1e-8)
        if img_denorm.ndim == 3:
            gray = rgb2gray(img_denorm)
        else:
            gray = img_denorm[:, :, 0]
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_points + 2))
        feats.append(hist.astype(np.float64) / (hist.sum() + 1e-8))
    return np.array(feats, dtype=np.float64)


def extract_all_features(X, verbose=True):
    """
    Extract and concatenate HOG + Color Histogram + LBP for each image.

    Args:
        X: (N, C, H, W) float32
    Returns:
        (N, total_dim) float64
    """
    if verbose:
        print("[Features] Extracting HOG...", flush=True)
    hog_f  = extract_hog(X)
    if verbose:
        print(f"           HOG dim = {hog_f.shape[1]}")
        print("[Features] Extracting Color Histogram...", flush=True)
    hist_f = extract_color_histogram(X)
    if verbose:
        print(f"           Histogram dim = {hist_f.shape[1]}")
        print("[Features] Extracting LBP...", flush=True)
    lbp_f  = extract_lbp(X)
    if verbose:
        print(f"           LBP dim = {lbp_f.shape[1]}")
    combined = np.concatenate([hog_f, hist_f, lbp_f], axis=1)
    if verbose:
        print(f"[Features] Total feature dim = {combined.shape[1]}")
    return combined


# ─────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────

def apply_pca(X_train, X_test, variance=None, scale=True):
    """
    Fit PCA on X_train, transform both sets.

    Args:
        variance: float (e.g. 0.95) or int (exact n_components)
        scale   : StandardScaler before PCA

    Returns:
        X_train_pca, X_test_pca, pca, scaler
    """
    if variance is None:
        variance = config.PCA_VARIANCE
    scaler = StandardScaler() if scale else None
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
    pca = PCA(n_components=variance, random_state=config.RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    print(f"[PCA] Variance={variance} → components={pca.n_components_} "
          f"| train={X_train_pca.shape} | test={X_test_pca.shape}")
    return X_train_pca, X_test_pca, pca, scaler


def pca_variance_curve(X_train, max_components=200, save_path=None):
    """
    Plot cumulative explained variance vs number of PCA components.
    Returns the array of explained variance ratios.
    """
    n_comp = min(max_components, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_comp, random_state=config.RANDOM_SEED)
    scaler = StandardScaler()
    pca.fit(scaler.fit_transform(X_train))
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar)+1), cumvar, color="#4C72B0", linewidth=2)
    for thr in [0.80, 0.90, 0.95, 0.99]:
        idx = np.searchsorted(cumvar, thr)
        ax.axhline(thr, linestyle="--", color="gray", alpha=0.5)
        ax.axvline(idx, linestyle="--", color="gray", alpha=0.5)
        ax.text(idx+1, thr-0.02, f"{int(thr*100)}% ({idx} comp.)", fontsize=8)
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA — Explained Variance vs Components")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return cumvar


def plot_accuracy_vs_pca(results: dict, save_path=None):
    """
    Plot accuracy vs PCA variance retention for multiple models.

    Args:
        results: dict like {'KNN': [(var, acc), ...], 'NB': [...]}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for (name, pts), color in zip(results.items(), colors):
        variances, accs = zip(*pts)
        ax.plot([f"{int(v*100)}%" for v in variances], accs,
                marker="o", label=name, color=color, linewidth=2)
    ax.set_xlabel("PCA Variance Retained")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs PCA Compression")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────
# t-SNE Visualization
# ─────────────────────────────────────────────────────────────

def plot_tsne(X, y, title="t-SNE Embedding", max_samples=1000, save_path=None):
    """
    Plot 2D t-SNE embedding of feature vectors, colored by class.

    Args:
        X: (N, D) feature matrix
        y: (N,)  class labels
        max_samples: cap for speed
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    print("[t-SNE] Fitting... (may take ~30s)", flush=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # PCA to 50D first for speed
    pca_pre = PCA(n_components=min(50, Xs.shape[1]), random_state=config.RANDOM_SEED)
    Xs = pca_pre.fit_transform(Xs)
    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, perplexity=30, max_iter=1000)
    emb  = tsne.fit_transform(Xs)

    cmap  = plt.cm.get_cmap("tab10", config.NUM_CLASSES)
    names = config.CIFAR10_CLASSES

    fig, ax = plt.subplots(figsize=(10, 7))
    for cls in range(config.NUM_CLASSES):
        mask = y == cls
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   color=cmap(cls), label=names[cls],
                   alpha=0.6, s=15, edgecolors="none")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
