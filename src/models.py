"""
Model architectures for Urban Expansion Monitoring.

- Multispectral backbone adaptation (VGG16, ResNet50, EfficientNet-B0)
- Feature Pyramid Network (FPN)
- Classification head
- Siamese change-detection network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import NUM_CHANNELS, NUM_CLASSES, FPN_CHANNELS


# ═══════════════════════════════════════════════════════
#  Backbone Builders
# ═══════════════════════════════════════════════════════

def _expand_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Expand a pretrained 3-channel conv to `in_channels` channels.
    Extra channel weights are initialized by averaging the original RGB weights.
    """
    out_c = conv.out_channels
    k = conv.kernel_size
    s = conv.stride
    p = conv.padding

    new_conv = nn.Conv2d(in_channels, out_c, k, stride=s, padding=p, bias=(conv.bias is not None))
    with torch.no_grad():
        # Copy original RGB weights
        new_conv.weight[:, :3] = conv.weight[:, :3]
        # Extra channels: average of RGB weights
        avg = conv.weight[:, :3].mean(dim=1, keepdim=True)
        for c in range(3, in_channels):
            new_conv.weight[:, c:c + 1] = avg
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def build_vgg16(in_channels=NUM_CHANNELS, pretrained=True):
    """VGG16 backbone returning multi-scale feature maps."""
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    base = models.vgg16(weights=weights)
    # Replace first conv
    base.features[0] = _expand_first_conv(base.features[0], in_channels)

    # Split into blocks for FPN taps
    # VGG16 feature blocks end at pool layers: indices 4, 9, 16, 23, 30
    blocks = nn.ModuleList([
        base.features[:10],   # through pool2  -> stride 4
        base.features[10:17], # through pool3  -> stride 8
        base.features[17:24], # through pool4  -> stride 16
    ])
    channels = [128, 256, 512]
    return blocks, channels


def build_resnet50(in_channels=NUM_CHANNELS, pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    base = models.resnet50(weights=weights)
    base.conv1 = _expand_first_conv(base.conv1, in_channels)

    # layer1 -> stride 4, layer2 -> stride 8, layer3 -> stride 16
    blocks = nn.ModuleList([
        nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1),
        base.layer2,
        base.layer3,
    ])
    channels = [256, 512, 1024]
    return blocks, channels


def build_efficientnet_b0(in_channels=NUM_CHANNELS, pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    base = models.efficientnet_b0(weights=weights)
    base.features[0][0] = _expand_first_conv(base.features[0][0], in_channels)

    # EfficientNet-B0 features: output channels are determined by last block in each group
    # features[0:3] -> out 24, features[3:5] -> out 80, features[5:7] -> out 192
    blocks = nn.ModuleList([
        nn.Sequential(*base.features[:3]),  # stride 4, out=24
        nn.Sequential(*base.features[3:5]), # stride 8, out=80
        nn.Sequential(*base.features[5:7]), # stride 16, out=192
    ])
    channels = [24, 80, 192]
    return blocks, channels


def build_mobilenet_v3_small(in_channels=NUM_CHANNELS, pretrained=True):
    """MobileNetV3-Small backbone — lightweight for edge deployment (Pillar V)."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    base = models.mobilenet_v3_small(weights=weights)
    base.features[0][0] = _expand_first_conv(base.features[0][0], in_channels)

    # MobileNetV3-Small features split for FPN:
    # features[0:4]  -> stride 4, out=24
    # features[4:9]  -> stride 8, out=48
    # features[9:13] -> stride 16, out=96
    blocks = nn.ModuleList([
        nn.Sequential(*base.features[:4]),
        nn.Sequential(*base.features[4:9]),
        nn.Sequential(*base.features[9:]),
    ])
    # Probe output channels
    with torch.no_grad():
        dummy = torch.randn(1, in_channels, 64, 64)
        channels = []
        out = dummy
        for block in blocks:
            out = block(out)
            channels.append(out.shape[1])
    return blocks, channels


def build_swin_tiny(in_channels=NUM_CHANNELS, pretrained=True):
    """Swin Transformer-Tiny backbone — ViT comparison for reviewers."""
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        in_chans=in_channels,
        features_only=True,
        out_indices=(1, 2, 3),  # stride 4, 8, 16
    )
    # Probe output channels
    with torch.no_grad():
        dummy = torch.randn(1, in_channels, 224, 224)
        feats = model(dummy)
        channels = [f.shape[1] for f in feats]

    class SwinBlocks(nn.Module):
        def __init__(self, swin_model):
            super().__init__()
            self.model = swin_model

        def forward(self, x):
            # timm features_only returns list of feature maps
            return self.model(x)

    wrapper = SwinBlocks(model)
    # Return as a single module (not ModuleList) — UrbanClassifier handles via extract_features
    return wrapper, channels


def build_convnext_tiny(in_channels=NUM_CHANNELS, pretrained=True):
    """ConvNeXt-Tiny backbone — modern CNN SOTA, proves backbone-agnostic framework."""
    model = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        in_chans=in_channels,
        features_only=True,
        out_indices=(1, 2, 3),  # stride 4, 8, 16
    )
    with torch.no_grad():
        dummy = torch.randn(1, in_channels, 224, 224)
        feats = model(dummy)
        channels = [f.shape[1] for f in feats]

    class ConvNeXtBlocks(nn.Module):
        def __init__(self, cnext_model):
            super().__init__()
            self.model = cnext_model

        def forward(self, x):
            return self.model(x)

    wrapper = ConvNeXtBlocks(model)
    return wrapper, channels


def build_prithvi(in_channels=NUM_CHANNELS, pretrained=True):
    """
    Prithvi-inspired ViT backbone for geospatial imagery.
    Uses timm's ViT with 6-channel input to approximate Prithvi's architecture.
    NASA/IBM Prithvi-100M is a ViT-Large pretrained on HLS satellite data.
    We use ViT-Small here for RTX 4070 VRAM compatibility (~4GB).
    """
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        in_chans=in_channels,
        features_only=False,
        num_classes=0,  # remove classifier, get features
    )

    class PrithviBlocks(nn.Module):
        """Wraps ViT to produce multi-scale features via intermediate blocks."""
        def __init__(self, vit_model):
            super().__init__()
            self.vit = vit_model
            self.embed_dim = vit_model.embed_dim  # 384 for vit_small
            # Project intermediate ViT blocks to different "scales"
            # ViT doesn't have natural multi-scale, so we use blocks 4, 8, 12
            # and reshape patch tokens back to spatial maps
            self.proj1 = nn.Conv2d(self.embed_dim, self.embed_dim // 2, 1)
            self.proj2 = nn.Conv2d(self.embed_dim, self.embed_dim, 1)
            self.proj3 = nn.Conv2d(self.embed_dim, self.embed_dim * 2, 1)

        def _get_intermediate_features(self, x):
            B = x.shape[0]
            x = self.vit.patch_embed(x)
            if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
                cls_token = self.vit.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
            x = x + self.vit.pos_embed
            x = self.vit.pos_drop(x) if hasattr(self.vit, 'pos_drop') else x
            x = self.vit.patch_drop(x) if hasattr(self.vit, 'patch_drop') else x
            x = self.vit.norm_pre(x) if hasattr(self.vit, 'norm_pre') else x

            features = []
            num_blocks = len(self.vit.blocks)
            tap_indices = [num_blocks // 3 - 1, 2 * num_blocks // 3 - 1, num_blocks - 1]

            for i, block in enumerate(self.vit.blocks):
                x = block(x)
                if i in tap_indices:
                    features.append(x)
            return features

        def forward(self, x):
            B, C, H, W = x.shape
            feats = self._get_intermediate_features(x)
            # Calculate spatial dims from patch size
            patch_size = self.vit.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            h = H // patch_size
            w = W // patch_size

            results = []
            for i, (f, proj) in enumerate(zip(feats, [self.proj1, self.proj2, self.proj3])):
                # Remove CLS token if present
                if f.shape[1] == h * w + 1:
                    f = f[:, 1:]
                spatial = f.transpose(1, 2).reshape(B, self.embed_dim, h, w)
                # Apply scale projection
                projected = proj(spatial)
                # Downsample to simulate multi-scale (stride 4, 8, 16)
                if i == 0:
                    projected = projected  # "stride 4" equivalent
                elif i == 1:
                    projected = F.avg_pool2d(projected, 2)  # "stride 8"
                else:
                    projected = F.avg_pool2d(projected, 4)  # "stride 16"
                results.append(projected)
            return results

    wrapper = PrithviBlocks(model)
    channels = [model.embed_dim // 2, model.embed_dim, model.embed_dim * 2]  # [192, 384, 768]
    return wrapper, channels


BACKBONE_BUILDERS = {
    "vgg16": build_vgg16,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "mobilenet_v3_small": build_mobilenet_v3_small,
    "swin_tiny": build_swin_tiny,
    "convnext_tiny": build_convnext_tiny,
    "prithvi": build_prithvi,
}


# ═══════════════════════════════════════════════════════
#  Feature Pyramid Network
# ═══════════════════════════════════════════════════════

class FPN(nn.Module):
    """3-level Feature Pyramid Network (Lin et al., 2017)."""

    def __init__(self, in_channels_list, out_channels=FPN_CHANNELS):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()
        for in_c in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))
            self.smooth_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, features):
        """
        Args:
            features: list of [C4, C8, C16] feature maps (low-to-high stride)
        Returns:
            list of FPN feature maps (same spatial sizes as inputs)
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear",
                               align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        out = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]
        return out


# ═══════════════════════════════════════════════════════
#  Classification Head
# ═══════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """GAP -> Dense(512) -> Dense(256) -> Dense(num_classes)"""

    def __init__(self, in_features, num_classes=NUM_CLASSES):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ═══════════════════════════════════════════════════════
#  Full Classification Model (Backbone + FPN + Head)
# ═══════════════════════════════════════════════════════

class UrbanClassifier(nn.Module):
    """
    Multispectral urban classifier with FPN multi-scale fusion.
    """

    # Models that require fixed input sizes (transformers with position embeddings)
    _FIXED_SIZE_MODELS = {"swin_tiny": 224, "prithvi": 224}

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        builder = BACKBONE_BUILDERS[backbone_name]
        self.blocks, ch_list = builder(pretrained=pretrained)
        self._is_wrapper = not isinstance(self.blocks, nn.ModuleList)
        self._required_size = self._FIXED_SIZE_MODELS.get(backbone_name, None)
        self.fpn = FPN(ch_list, FPN_CHANNELS)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Concatenate pooled features from all 3 FPN levels
        self.head = ClassificationHead(FPN_CHANNELS * 3, NUM_CLASSES)

    def extract_features(self, x):
        # Resize if backbone requires fixed input size (Swin, Prithvi)
        if self._required_size and x.shape[-1] != self._required_size:
            x = F.interpolate(x, size=(self._required_size, self._required_size),
                              mode="bilinear", align_corners=False)
        if self._is_wrapper:
            # Swin, ConvNeXt, Prithvi — wrapper returns list of features directly
            return self.blocks(x)
        features = []
        out = x
        for block in self.blocks:
            out = block(out)
            features.append(out)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        fpn_out = self.fpn(features)
        # Global average pool each FPN level and concatenate
        pooled = [self.gap(f).flatten(1) for f in fpn_out]
        combined = torch.cat(pooled, dim=1)
        logits = self.head(combined)
        return logits

    def get_feature_vector(self, x):
        """Return feature vector (for Siamese use)."""
        features = self.extract_features(x)
        fpn_out = self.fpn(features)
        pooled = [self.gap(f).flatten(1) for f in fpn_out]
        return torch.cat(pooled, dim=1)


# ═══════════════════════════════════════════════════════
#  Siamese Change Detection Network
# ═══════════════════════════════════════════════════════

class SiameseChangeDetector(nn.Module):
    """
    Siamese network for bi-temporal urban expansion change detection.
    Shared-weight encoder produces feature vectors for two dates;
    concatenated features are classified as change / no-change.
    """

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True):
        super().__init__()
        self.encoder = UrbanClassifier(backbone_name, pretrained)
        # Replace the classification head with a change head
        feat_dim = FPN_CHANNELS * 3
        self.change_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # change / no-change
        )

    def forward(self, x1, x2):
        f1 = self.encoder.get_feature_vector(x1)
        f2 = self.encoder.get_feature_vector(x2)
        combined = torch.cat([f1, f2], dim=1)
        return self.change_head(combined)


# ═══════════════════════════════════════════════════════
#  Progressive Unfreezing Helpers
# ═══════════════════════════════════════════════════════

def freeze_backbone(model: UrbanClassifier):
    """Freeze all backbone parameters (Stage 1)."""
    for p in model.blocks.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(model: UrbanClassifier):
    """Unfreeze the last backbone block (Stage 2)."""
    if model._is_wrapper:
        # For wrapper models, unfreeze last ~25% of parameters
        params = list(model.blocks.parameters())
        cutoff = len(params) * 3 // 4
        for p in params[cutoff:]:
            p.requires_grad = True
    else:
        for p in model.blocks[-1].parameters():
            p.requires_grad = True


def unfreeze_all(model: UrbanClassifier):
    """Unfreeze everything (Stage 3)."""
    for p in model.parameters():
        p.requires_grad = True


UNFREEZE_FNS = {
    "head": freeze_backbone,
    "last_blocks": unfreeze_last_blocks,
    "all": unfreeze_all,
}


if __name__ == "__main__":
    # Quick test all backbones
    for name in BACKBONE_BUILDERS:
        try:
            # Swin/ViT need 224, others work with 256
            img_size = 224 if name in ("swin_tiny", "prithvi") else 256
            model = UrbanClassifier(name, pretrained=False)
            x = torch.randn(2, NUM_CHANNELS, img_size, img_size)
            out = model(x)
            total_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"{name}: output={out.shape}, params={total_params:.1f}M")
        except Exception as e:
            print(f"{name}: FAILED - {e}")

    # Siamese test
    siam = SiameseChangeDetector("efficientnet_b0", pretrained=False)
    x1 = torch.randn(2, NUM_CHANNELS, 256, 256)
    x2 = torch.randn(2, NUM_CHANNELS, 256, 256)
    print(f"Siamese output: {siam(x1, x2).shape}")
