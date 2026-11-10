
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss for 3D medical image segmentation.

    Background class is excluded from Dice calculation.

    Args:
        num_classes (int): Number of segmentation classes.
        eps (float): Smoothing factor to prevent division by zero.
    """

    def __init__(self, num_classes=4, eps=1e-6):
        super().__init__()

        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):

        # Apply softmax over the class dimension
        probabilities = torch.softmax(logits, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes
        )

        # Rearrange dimensions to (B, C, D, H, W)
        targets_one_hot = targets_one_hot.permute(
            0, 4, 1, 2, 3
        ).float()

        # Exclude background class
        probabilities = probabilities[:, 1:]
        targets_one_hot = targets_one_hot[:, 1:]

        # Calculate Dice over batch and spatial dimensions
        dimensions = (0, 2, 3, 4)

        intersection = (
            probabilities * targets_one_hot
        ).sum(dim=dimensions)

        denominator = (
            probabilities.sum(dim=dimensions)
            + targets_one_hot.sum(dim=dimensions)
        )

        dice = (
            2.0 * intersection + self.eps
        ) / (
            denominator + self.eps
        )

        dice_loss = 1.0 - dice.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss.

    Total Loss = CE Loss + Dice Loss

    Args:
        num_classes (int): Number of segmentation classes.
        ce_weight (float): Weight assigned to Cross-Entropy Loss.
        dice_weight (float): Weight assigned to Dice Loss.
    """

    def __init__(
        self,
        num_classes=4,
        ce_weight=1.0,
        dice_weight=1.0
    ):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(
            num_classes=num_classes
        )

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        ce = self.ce_loss(
            logits,
            targets
        )

        dice = self.dice_loss(
            logits,
            targets
        )

        total_loss = (
            self.ce_weight * ce
            + self.dice_weight * dice
        )

        return total_loss, ce, dice
