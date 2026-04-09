"""
Loss functions: Weighted CE + Focal Loss + Dice Loss.
L_total = 0.6 * L_CE + 0.3 * L_FL(gamma=2) + 0.1 * L_Dice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import LOSS_WEIGHTS, FOCAL_GAMMA, NUM_CLASSES


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for class imbalance."""

    def __init__(self, gamma=FOCAL_GAMMA, reduction="mean", class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights  # tensor of per-class weights

    def forward(self, logits, targets):
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class classification."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes=probs.shape[1]).float()

        intersection = (probs * targets_oh).sum(dim=0)
        union = probs.sum(dim=0) + targets_oh.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    L_total = w_ce * CrossEntropy + w_focal * FocalLoss + w_dice * DiceLoss

    Supports per-class weights to handle imbalanced datasets (e.g., Transition class).
    """

    def __init__(self, w_ce=LOSS_WEIGHTS["ce"], w_focal=LOSS_WEIGHTS["focal"],
                 w_dice=LOSS_WEIGHTS["dice"], focal_gamma=FOCAL_GAMMA,
                 class_weights=None):
        super().__init__()
        self.w_ce = w_ce
        self.w_focal = w_focal
        self.w_dice = w_dice
        # class_weights: e.g., [1.0, 1.0, 3.0] to upweight Transition
        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=cw)
        self.focal = FocalLoss(gamma=focal_gamma, class_weights=cw)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        l_ce = self.ce(logits, targets)
        l_fl = self.focal(logits, targets)
        l_dice = self.dice(logits, targets)
        return self.w_ce * l_ce + self.w_focal * l_fl + self.w_dice * l_dice


class ChangeLoss(nn.Module):
    """Binary cross-entropy loss for Siamese change detection."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets)
