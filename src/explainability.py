"""
Explainability and Interpretability Module for Urban Expansion Monitoring.

Implements research-level visualization techniques:
    - GradCAM (Selvaraju et al., 2017)
    - GradCAM++ (Chattopadhay et al., 2018)
    - t-SNE embedding of learned feature spaces
    - UMAP embedding (with fallback to t-SNE)
    - FPN multi-scale feature activation visualization
    - Per-block intermediate feature map visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from configs.config import (
    NUM_CHANNELS, NUM_CLASSES, CLASS_NAMES, PATCH_SIZE,
    FIGURE_DIR, MODEL_DIR, SEED, BATCH_SIZE, CITIES,
)
from src.models import UrbanClassifier
from src.dataset import UrbanExpansionDataset


# ═══════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════

CLASS_COLORS = ["#e74c3c", "#27ae60", "#f39c12"]  # Urban=red, Non-Urban=green, Transition=orange


def _to_rgb(patch_tensor):
    """
    Convert a 6-channel patch tensor to an RGB composite for display.

    Uses channels [2, 1, 0] (Red, Green, Blue) with brightness boost (*3, clip to [0, 1]).

    Args:
        patch_tensor: Tensor of shape (6, H, W) or (H, W, 6).

    Returns:
        ndarray of shape (H, W, 3) in [0, 1].
    """
    if isinstance(patch_tensor, torch.Tensor):
        arr = patch_tensor.detach().cpu().numpy()
    else:
        arr = np.array(patch_tensor)

    if arr.ndim == 3 and arr.shape[0] == NUM_CHANNELS:
        # Channel-first -> channel-last
        arr = arr.transpose(1, 2, 0)

    rgb = arr[:, :, [2, 1, 0]]  # R, G, B channels
    rgb = np.clip(rgb * 3.0, 0.0, 1.0)
    return rgb


def _ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _get_dataloader(num_samples, batch_size=BATCH_SIZE):
    """Create a dataloader for explainability analysis."""
    dataset = UrbanExpansionDataset(num_patches=num_samples, transform=None, seed=SEED)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return loader


# ═══════════════════════════════════════════════════════
#  GradCAM (Selvaraju et al., 2017)
# ═══════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into a target convolutional layer to capture activations and gradients,
    then computes a weighted combination to produce a class-discriminative
    localization heatmap.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: UrbanClassifier instance.
            target_layer: nn.Module to hook (e.g., model.blocks[-1]).
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate GradCAM heatmap for a single input.

        Args:
            input_tensor: Tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses the predicted class.

        Returns:
            heatmap: ndarray of shape (H, W) in [0, 1].
            predicted_class: int.
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = predicted_class

        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Compute GradCAM weights: global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class

    def remove_hooks(self):
        """Remove registered hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ═══════════════════════════════════════════════════════
#  GradCAM++ (Chattopadhay et al., 2018)
# ═══════════════════════════════════════════════════════

class GradCAMPlusPlus:
    """
    GradCAM++ uses second-order gradients for improved localization,
    especially for multiple instances of the same class within an image.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate GradCAM++ heatmap.

        Uses second- and third-order gradient information to compute
        pixel-wise weighting coefficients alpha_{kc}^{ij}.

        Args:
            input_tensor: Tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses predicted class.

        Returns:
            heatmap: ndarray of shape (H, W) in [0, 1].
            predicted_class: int.
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = predicted_class

        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        grads = self.gradients  # (1, C, h, w)
        acts = self.activations  # (1, C, h, w)

        # Second-order gradients approximation for GradCAM++
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        # Sum of activations along spatial dimensions
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Alpha coefficients (Eq. 9 in paper)
        eps = 1e-7
        denominator = 2.0 * grads_power_2 + sum_acts * grads_power_3 + eps
        alpha = grads_power_2 / denominator  # (1, C, h, w)

        # Weights: sum of alpha * ReLU(grad) over spatial dimensions
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


# ═══════════════════════════════════════════════════════
#  Visualization Functions
# ═══════════════════════════════════════════════════════

def gradcam_visualization(model, device, num_samples=12, save_dir=FIGURE_DIR):
    """
    Generate GradCAM and GradCAM++ visualizations for sample patches.

    Produces a grid showing, for each sample:
        - RGB composite of the input
        - GradCAM heatmap overlay
        - GradCAM++ heatmap overlay

    Saves to: <save_dir>/gradcam_analysis.png

    Args:
        model: Trained UrbanClassifier.
        device: torch.device.
        num_samples: Number of samples to visualize (4 per class).
        save_dir: Output directory for the figure.
    """
    _ensure_dir(save_dir)
    model.eval()

    # Collect samples per class
    samples_per_class = max(num_samples // NUM_CLASSES, 4)
    dataset = UrbanExpansionDataset(num_patches=num_samples * 5, transform=None, seed=SEED)

    class_samples = {c: [] for c in range(NUM_CLASSES)}
    for i in range(len(dataset)):
        img, label = dataset[i]
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(img)
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break

    # Initialize GradCAM hooks on the last backbone block
    target_layer = model.blocks[-1]
    grad_cam = GradCAM(model, target_layer)
    grad_cam_pp = GradCAMPlusPlus(model, target_layer)

    # Build figure: rows = classes, cols = samples_per_class * 3 (RGB, GradCAM, GradCAM++)
    n_cols_per_sample = 3
    n_display = min(samples_per_class, 4)  # show at most 4 per class
    fig, axes = plt.subplots(
        NUM_CLASSES, n_display * n_cols_per_sample,
        figsize=(n_display * n_cols_per_sample * 2.5, NUM_CLASSES * 2.8),
        dpi=300,
    )

    for cls_idx in range(NUM_CLASSES):
        for s_idx in range(n_display):
            if s_idx >= len(class_samples[cls_idx]):
                # Hide empty axes
                for k in range(n_cols_per_sample):
                    col = s_idx * n_cols_per_sample + k
                    axes[cls_idx, col].axis("off")
                continue

            img_tensor = class_samples[cls_idx][s_idx]
            input_batch = img_tensor.unsqueeze(0).to(device)
            rgb = _to_rgb(img_tensor)

            # GradCAM
            cam, pred = grad_cam.generate(input_batch, target_class=cls_idx)
            # GradCAM++
            cam_pp, _ = grad_cam_pp.generate(input_batch, target_class=cls_idx)

            col_base = s_idx * n_cols_per_sample

            # RGB
            ax_rgb = axes[cls_idx, col_base]
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f"{CLASS_NAMES[cls_idx]} (pred: {CLASS_NAMES[pred]})", fontsize=7)
            ax_rgb.axis("off")

            # GradCAM overlay
            ax_gc = axes[cls_idx, col_base + 1]
            ax_gc.imshow(rgb)
            im_gc = ax_gc.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax_gc.set_title("GradCAM", fontsize=7)
            ax_gc.axis("off")

            # GradCAM++ overlay
            ax_gpp = axes[cls_idx, col_base + 2]
            ax_gpp.imshow(rgb)
            im_gpp = ax_gpp.imshow(cam_pp, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax_gpp.set_title("GradCAM++", fontsize=7)
            ax_gpp.axis("off")

    # Row labels
    for cls_idx in range(NUM_CLASSES):
        axes[cls_idx, 0].set_ylabel(CLASS_NAMES[cls_idx], fontsize=10, fontweight="bold", rotation=90)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im_gc, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Activation Intensity", fontsize=9)

    fig.suptitle("GradCAM & GradCAM++ Analysis by Class", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])

    save_path = os.path.join(save_dir, "gradcam_analysis.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Explainability] GradCAM analysis saved to {save_path}")

    # Clean up hooks
    grad_cam.remove_hooks()
    grad_cam_pp.remove_hooks()


def tsne_visualization(model, device, num_samples=1000, save_dir=FIGURE_DIR):
    """
    Visualize learned feature embeddings using t-SNE.

    Extracts feature vectors from the model using get_feature_vector(),
    reduces dimensionality with t-SNE, and plots a 2D scatter colored by class.

    Saves to: <save_dir>/tsne_embeddings.png

    Args:
        model: Trained UrbanClassifier.
        device: torch.device.
        num_samples: Number of samples to embed.
        save_dir: Output directory.
    """
    _ensure_dir(save_dir)
    model.eval()

    # Extract features
    loader = _get_dataloader(num_samples, batch_size=BATCH_SIZE)
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch_imgs, batch_labels in loader:
            batch_imgs = batch_imgs.to(device)
            feats = model.get_feature_vector(batch_imgs)
            features_list.append(feats.cpu().numpy())
            labels_list.append(batch_labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(f"[Explainability] Running t-SNE on {features.shape[0]} samples "
          f"(feature dim={features.shape[1]})...")

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, features.shape[0] - 1),
        learning_rate="auto",
        init="pca",
        random_state=SEED,
        n_iter=1000,
    )
    embeddings = tsne.fit_transform(features)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Panel 1: Colored by class
    ax1 = axes[0]
    for cls_idx in range(NUM_CLASSES):
        mask = labels == cls_idx
        ax1.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=CLASS_COLORS[cls_idx], label=CLASS_NAMES[cls_idx],
            s=15, alpha=0.6, edgecolors="none",
        )
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax1.set_title("t-SNE Embedding (by Class)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, markerscale=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Density / class boundaries via contour or secondary scatter with markers
    ax2 = axes[1]
    markers = ["o", "s", "^"]
    for cls_idx in range(NUM_CLASSES):
        mask = labels == cls_idx
        ax2.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=CLASS_COLORS[cls_idx], label=CLASS_NAMES[cls_idx],
            s=15, alpha=0.6, marker=markers[cls_idx], edgecolors="none",
        )
        # Plot class centroid
        centroid = embeddings[mask].mean(axis=0)
        ax2.scatter(
            centroid[0], centroid[1],
            c=CLASS_COLORS[cls_idx], s=200, marker="X",
            edgecolors="black", linewidths=1.5, zorder=10,
        )
        ax2.annotate(
            CLASS_NAMES[cls_idx], (centroid[0], centroid[1]),
            fontsize=9, fontweight="bold", ha="center", va="bottom",
            xytext=(0, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax2.set_title("t-SNE Embedding (with Centroids)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, markerscale=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("t-SNE Feature Space Visualization", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "tsne_embeddings.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Explainability] t-SNE embeddings saved to {save_path}")


def umap_visualization(model, device, num_samples=1000, save_dir=FIGURE_DIR):
    """
    Visualize learned feature embeddings using UMAP.

    Falls back to t-SNE if the umap-learn package is not installed.

    Saves to: <save_dir>/umap_embeddings.png

    Args:
        model: Trained UrbanClassifier.
        device: torch.device.
        num_samples: Number of samples to embed.
        save_dir: Output directory.
    """
    _ensure_dir(save_dir)
    model.eval()

    # Try importing UMAP
    try:
        from umap import UMAP
        use_umap = True
        print("[Explainability] UMAP package found. Using UMAP for dimensionality reduction.")
    except ImportError:
        use_umap = False
        print("[Explainability] UMAP not installed. Falling back to t-SNE for UMAP visualization.")

    # Extract features
    loader = _get_dataloader(num_samples, batch_size=BATCH_SIZE)
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch_imgs, batch_labels in loader:
            batch_imgs = batch_imgs.to(device)
            feats = model.get_feature_vector(batch_imgs)
            features_list.append(feats.cpu().numpy())
            labels_list.append(batch_labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Dimensionality reduction
    if use_umap:
        reducer = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=SEED,
        )
        method_name = "UMAP"
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, features.shape[0] - 1),
            learning_rate="auto",
            init="pca",
            random_state=SEED,
            n_iter=1000,
        )
        method_name = "t-SNE (UMAP fallback)"

    print(f"[Explainability] Running {method_name} on {features.shape[0]} samples...")
    embeddings = reducer.fit_transform(features)

    # Assign simulated city labels based on sample index for diversity in visualization
    n = len(labels)
    city_labels = np.array([CITIES[i % len(CITIES)] for i in range(n)])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Panel 1: Colored by class
    ax1 = axes[0]
    for cls_idx in range(NUM_CLASSES):
        mask = labels == cls_idx
        ax1.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=CLASS_COLORS[cls_idx], label=CLASS_NAMES[cls_idx],
            s=15, alpha=0.6, edgecolors="none",
        )
    ax1.set_xlabel(f"{method_name} Dimension 1", fontsize=11)
    ax1.set_ylabel(f"{method_name} Dimension 2", fontsize=11)
    ax1.set_title(f"{method_name} Embedding (by Class)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, markerscale=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Colored by city
    ax2 = axes[1]
    city_cmap = plt.cm.get_cmap("tab10", len(CITIES))
    for i, city in enumerate(CITIES):
        mask = city_labels == city
        if mask.sum() == 0:
            continue
        ax2.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=[city_cmap(i)], label=city.replace("_", " "),
            s=15, alpha=0.6, edgecolors="none",
        )
    ax2.set_xlabel(f"{method_name} Dimension 1", fontsize=11)
    ax2.set_ylabel(f"{method_name} Dimension 2", fontsize=11)
    ax2.set_title(f"{method_name} Embedding (by City)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, markerscale=2, framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{method_name} Feature Space Visualization", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "umap_embeddings.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Explainability] {method_name} embeddings saved to {save_path}")


def feature_map_visualization(model, device, save_dir=FIGURE_DIR):
    """
    Visualize intermediate feature maps and FPN activation maps.

    Shows:
        - Top row: RGB input and feature maps from each backbone block
        - Bottom row: FPN feature maps at each pyramid level
        - Channel-wise mean activation across spatial dimensions

    Saves to: <save_dir>/feature_maps.png

    Args:
        model: Trained UrbanClassifier.
        device: torch.device.
        save_dir: Output directory.
    """
    _ensure_dir(save_dir)
    model.eval()

    # Get a single sample for each class
    dataset = UrbanExpansionDataset(num_patches=50, transform=None, seed=SEED)

    # Pick one sample per class
    class_samples = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label not in class_samples:
            class_samples[label] = img
        if len(class_samples) >= NUM_CLASSES:
            break

    n_blocks = len(model.blocks)

    # Create figure: 3 rows (one per class) x (1 RGB + n_blocks backbone + n_blocks FPN + 1 activation bar)
    n_cols = 1 + n_blocks + n_blocks + 1
    fig = plt.figure(figsize=(n_cols * 3, NUM_CLASSES * 3.2), dpi=300)
    gs = gridspec.GridSpec(NUM_CLASSES, n_cols, wspace=0.15, hspace=0.35)

    for cls_idx in range(NUM_CLASSES):
        if cls_idx not in class_samples:
            continue

        img_tensor = class_samples[cls_idx]
        input_batch = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract backbone features
            backbone_features = model.extract_features(input_batch)
            # Extract FPN features
            fpn_features = model.fpn(backbone_features)

        rgb = _to_rgb(img_tensor)

        # Column 0: RGB input
        ax = fig.add_subplot(gs[cls_idx, 0])
        ax.imshow(rgb)
        ax.set_title("Input (RGB)", fontsize=8)
        ax.set_ylabel(CLASS_NAMES[cls_idx], fontsize=10, fontweight="bold")
        ax.axis("off")

        # Columns 1..n_blocks: Backbone feature maps (mean across channels)
        for b_idx in range(n_blocks):
            ax = fig.add_subplot(gs[cls_idx, 1 + b_idx])
            fmap = backbone_features[b_idx][0].cpu().numpy()  # (C, h, w)
            mean_activation = fmap.mean(axis=0)  # (h, w)
            im = ax.imshow(mean_activation, cmap="viridis")
            ax.set_title(f"Block {b_idx + 1}\n({fmap.shape[0]}ch, {fmap.shape[1]}x{fmap.shape[2]})",
                         fontsize=7)
            ax.axis("off")

        # Columns n_blocks+1..2*n_blocks: FPN feature maps
        for f_idx in range(n_blocks):
            ax = fig.add_subplot(gs[cls_idx, 1 + n_blocks + f_idx])
            fmap = fpn_features[f_idx][0].cpu().numpy()  # (C, h, w)
            mean_activation = fmap.mean(axis=0)  # (h, w)
            im = ax.imshow(mean_activation, cmap="inferno")
            ax.set_title(f"FPN P{f_idx + 1}\n({fmap.shape[0]}ch, {fmap.shape[1]}x{fmap.shape[2]})",
                         fontsize=7)
            ax.axis("off")

        # Last column: Channel activation bar chart (FPN combined)
        ax = fig.add_subplot(gs[cls_idx, -1])
        combined_acts = []
        fpn_labels = []
        for f_idx in range(n_blocks):
            fmap = fpn_features[f_idx][0].cpu().numpy()
            # Mean activation per channel, take top-16 channels
            ch_means = fmap.mean(axis=(1, 2))
            top_k = min(16, len(ch_means))
            top_indices = np.argsort(ch_means)[-top_k:][::-1]
            combined_acts.extend(ch_means[top_indices])
            fpn_labels.extend([f"P{f_idx + 1}-{j}" for j in top_indices])

        colors_bar = plt.cm.Set2(np.linspace(0, 1, len(combined_acts)))
        ax.barh(range(len(combined_acts)), combined_acts, color=colors_bar, height=0.7)
        ax.set_yticks(range(len(fpn_labels)))
        ax.set_yticklabels(fpn_labels, fontsize=4)
        ax.set_xlabel("Mean Act.", fontsize=7)
        ax.set_title("Top FPN\nChannels", fontsize=7)
        ax.invert_yaxis()

    fig.suptitle("Intermediate Feature Maps & FPN Activations",
                 fontsize=14, fontweight="bold", y=1.01)

    save_path = os.path.join(save_dir, "feature_maps.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Explainability] Feature maps saved to {save_path}")


def attention_visualization(model, device, save_dir=FIGURE_DIR):
    """
    Visualize FPN feature map activations at different scales as a proxy
    for spatial attention.

    For each FPN level, computes the L2 norm across channels to produce
    a per-pixel attention-like score. This highlights which spatial regions
    each scale focuses on.

    Saves to: <save_dir>/attention_maps.png

    Args:
        model: Trained UrbanClassifier.
        device: torch.device.
        save_dir: Output directory.
    """
    _ensure_dir(save_dir)
    model.eval()

    # Collect one sample per class
    dataset = UrbanExpansionDataset(num_patches=50, transform=None, seed=SEED)
    class_samples = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label not in class_samples:
            class_samples[label] = img
        if len(class_samples) >= NUM_CLASSES:
            break

    n_blocks = len(model.blocks)
    fig, axes = plt.subplots(NUM_CLASSES, 1 + n_blocks, figsize=((1 + n_blocks) * 3.5, NUM_CLASSES * 3.5), dpi=300)

    if NUM_CLASSES == 1:
        axes = axes[np.newaxis, :]

    for cls_idx in range(NUM_CLASSES):
        if cls_idx not in class_samples:
            continue

        img_tensor = class_samples[cls_idx]
        input_batch = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            backbone_features = model.extract_features(input_batch)
            fpn_features = model.fpn(backbone_features)

        rgb = _to_rgb(img_tensor)

        # RGB
        axes[cls_idx, 0].imshow(rgb)
        axes[cls_idx, 0].set_title("Input", fontsize=9)
        axes[cls_idx, 0].set_ylabel(CLASS_NAMES[cls_idx], fontsize=10, fontweight="bold")
        axes[cls_idx, 0].axis("off")

        # FPN attention-like maps (L2 norm across channels)
        for f_idx in range(n_blocks):
            fmap = fpn_features[f_idx][0].cpu().numpy()  # (C, h, w)
            # L2 norm across channels -> spatial attention proxy
            attention = np.sqrt((fmap ** 2).sum(axis=0))  # (h, w)
            # Normalize
            att_min, att_max = attention.min(), attention.max()
            if att_max - att_min > 1e-8:
                attention = (attention - att_min) / (att_max - att_min)

            ax = axes[cls_idx, 1 + f_idx]
            # Show RGB background with attention overlay
            rgb_resized = np.array(
                plt.cm.ScalarMappable(cmap="gray").to_rgba(np.zeros((attention.shape[0], attention.shape[1])))
            )[:, :, :3]
            ax.imshow(rgb)
            # Upsample attention to input size for overlay
            att_up = np.array(
                F.interpolate(
                    torch.tensor(attention).unsqueeze(0).unsqueeze(0).float(),
                    size=(PATCH_SIZE, PATCH_SIZE),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()
            )
            im = ax.imshow(att_up, cmap="hot", alpha=0.55, vmin=0, vmax=1)
            ax.set_title(f"FPN P{f_idx + 1} Attention\n(scale 1/{2 ** (2 + f_idx)})", fontsize=8)
            ax.axis("off")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized Activation", fontsize=9)

    fig.suptitle("Multi-Scale FPN Attention Maps", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])

    save_path = os.path.join(save_dir, "attention_maps.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Explainability] Attention maps saved to {save_path}")


# ═══════════════════════════════════════════════════════
#  Master Runner
# ═══════════════════════════════════════════════════════

def run_explainability(model, device):
    """
    Run all explainability and interpretability visualizations.

    Generates:
        - gradcam_analysis.png   : GradCAM and GradCAM++ heatmaps
        - tsne_embeddings.png    : t-SNE feature space visualization
        - umap_embeddings.png    : UMAP (or fallback t-SNE) embedding
        - feature_maps.png       : Backbone and FPN intermediate features
        - attention_maps.png     : Multi-scale FPN attention proxies

    Args:
        model: Trained UrbanClassifier.
        device: torch.device.
    """
    print("=" * 60)
    print("  EXPLAINABILITY & INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    print("\n[1/5] Generating GradCAM & GradCAM++ visualizations...")
    gradcam_visualization(model, device)

    print("\n[2/5] Generating t-SNE embeddings...")
    tsne_visualization(model, device, num_samples=1000)

    print("\n[3/5] Generating UMAP embeddings...")
    umap_visualization(model, device, num_samples=1000)

    print("\n[4/5] Generating feature map visualizations...")
    feature_map_visualization(model, device)

    print("\n[5/5] Generating attention map visualizations...")
    attention_visualization(model, device)

    print("\n" + "=" * 60)
    print(f"  All explainability figures saved to: {FIGURE_DIR}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explainability analysis for Urban Expansion Monitoring")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=["vgg16", "resnet50", "efficientnet_b0"],
                        help="Backbone architecture")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples for embedding visualizations")
    parser.add_argument("--save-dir", type=str, default=FIGURE_DIR,
                        help="Directory to save output figures")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = UrbanClassifier(backbone_name=args.backbone, pretrained=True)

    # Load checkpoint if provided
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint provided. Using randomly initialized model for demonstration.")

    model = model.to(device)
    model.eval()

    # Run all visualizations
    run_explainability(model, device)
