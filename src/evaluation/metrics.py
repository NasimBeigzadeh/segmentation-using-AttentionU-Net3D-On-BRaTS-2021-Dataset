
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def get_regions(mask):
    """
    Convert BraTS segmentation labels into
    Whole Tumor (WT), Tumor Core (TC),
    and Enhancing Tumor (ET) regions.
    """

    wt = mask > 0
    tc = (mask == 1) | (mask == 3)
    et = mask == 3

    return {
        "WT": wt,
        "TC": tc,
        "ET": et
    }


def calculate_binary_metrics(pred, target):
    """
    Calculate binary segmentation metrics.
    """

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    if pred.sum() == 0 and target.sum() == 0:
        return None

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou)
    }


def calculate_hd95(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate the 95th percentile Hausdorff Distance.

    Spacing is expressed in voxel-axis order.
    The default assumes isotropic unit spacing.
    """

    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 or target.sum() == 0:
        return np.nan

    pred_surface = np.logical_xor(
        pred,
        binary_erosion(pred)
    )

    target_surface = np.logical_xor(
        target,
        binary_erosion(target)
    )

    if not pred_surface.any() or not target_surface.any():
        return np.nan

    distance_target = distance_transform_edt(
        ~target_surface,
        sampling=spacing
    )

    distance_pred = distance_transform_edt(
        ~pred_surface,
        sampling=spacing
    )

    distances_1 = distance_target[pred_surface]
    distances_2 = distance_pred[target_surface]

    all_distances = np.concatenate(
        [distances_1, distances_2]
    )

    return float(np.percentile(all_distances, 95))
