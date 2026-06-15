"""Live inference for the demo: load a trained checkpoint on CPU, classify a
6-channel patch, and produce an RGB preview + Grad-CAM overlay.

Design goals (CHANDICON demo):
  * Runs on CPU (no GPU assumed at the venue).
  * Offline: builds backbones with pretrained=False so nothing is downloaded;
    the trained state_dict is loaded from outputs/models/.
  * Models are loaded lazily and cached, so the first request is the only slow one.
"""
import io
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402

import app_paths as P

# Make the project's src/ importable.
if str(P.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(P.PROJECT_ROOT))

from src.models import UrbanClassifier  # noqa: E402

CLASS_NAMES = ["Urban", "Non-Urban", "Transition"]
PATCH_SIZE = 256
DEVICE = torch.device("cpu")

_CHECKPOINTS = {
    "resnet50": "resnet50_best.pth",
    "efficientnet_b0": "efficientnet_b0_best.pth",
    "mobilenet_v3_small": "mobilenet_v3_small_best.pth",
    "swin_tiny": "swin_tiny_best.pth",
}

_MODEL_CACHE = {}


def available_models():
    return [k for k, fn in _CHECKPOINTS.items() if (P.MODELS / fn).exists()]


def _load_model(backbone):
    if backbone in _MODEL_CACHE:
        return _MODEL_CACHE[backbone]
    fn = _CHECKPOINTS.get(backbone)
    if fn is None:
        raise ValueError(f"Unknown backbone '{backbone}'")
    ckpt = P.MODELS / fn
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = UrbanClassifier(backbone_name=backbone, pretrained=False)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        # older checkpoints / non-tensor payloads
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE[backbone] = model
    return model


def _to_tensor(patch):
    """patch: np.ndarray (C, H, W) float32 -> (1, C, 256, 256) tensor."""
    t = torch.from_numpy(np.asarray(patch, dtype=np.float32))
    if t.ndim == 3 and tuple(t.shape[-2:]) != (PATCH_SIZE, PATCH_SIZE):
        t = F.interpolate(t.unsqueeze(0), size=(PATCH_SIZE, PATCH_SIZE),
                          mode="bilinear", align_corners=False).squeeze(0)
    return t.unsqueeze(0)


def classify(patch, backbone="resnet50"):
    """Return predicted class + full probability distribution."""
    model = _load_model(backbone)
    x = _to_tensor(patch).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    pred = int(np.argmax(probs))
    return {
        "backbone": backbone,
        "predicted_class": pred,
        "predicted_label": CLASS_NAMES[pred],
        "confidence": round(probs[pred], 4),
        "probabilities": {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
    }


def rgb_array(patch):
    """6-channel reflectance patch -> (H, W, 3) uint8 natural-colour composite."""
    arr = np.asarray(patch, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] >= 3:
        arr = arr.transpose(1, 2, 0)
    rgb = arr[:, :, [2, 1, 0]]  # R, G, B (matches src/explainability._to_rgb)
    rgb = np.clip(rgb * 3.0, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def preview_png(patch):
    img = Image.fromarray(rgb_array(patch))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _GradCAM:
    def __init__(self, model, target_layer):
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()


def gradcam_png(patch, backbone="resnet50", target_class=None):
    """Grad-CAM heatmap blended over the RGB preview. Returns (png_bytes, info).

    Only available for CNN backbones with a `.blocks` ModuleList (ResNet50,
    EfficientNet-B0, MobileNetV3). Transformer wrappers return None.
    """
    model = _load_model(backbone)
    if getattr(model, "_is_wrapper", False):
        return None, {"available": False, "reason": "Grad-CAM not supported for transformer backbone"}

    target_layer = model.blocks[-1]
    cam_hook = _GradCAM(model, target_layer)

    x = _to_tensor(patch).to(DEVICE)
    x.requires_grad_(True)
    model.zero_grad()
    logits = model(x)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())
    logits[0, target_class].backward()

    grads = cam_hook.gradients          # (1, C, h, w)
    acts = cam_hook.activations         # (1, C, h, w)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    base = rgb_array(patch).astype(np.float32) / 255.0
    heat = cm.jet(cam)[:, :, :3]
    overlay = np.clip(0.55 * base + 0.45 * heat, 0, 1)
    img = Image.fromarray((overlay * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), {"available": True, "target_class": target_class,
                            "target_label": CLASS_NAMES[target_class]}
