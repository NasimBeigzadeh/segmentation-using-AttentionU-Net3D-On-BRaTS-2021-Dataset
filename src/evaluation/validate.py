
import numpy as np
import torch

from tqdm.auto import tqdm
from torch.amp import autocast

from .metrics import (
    get_regions,
    calculate_binary_metrics,
    calculate_hd95
)


def validate_model(model, val_loader, criterion, device):

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0

    num_batches = 0

    region_results = {
        region: {
            "dice": [],
            "precision": [],
            "recall": [],
            "iou": [],
            "hd95": []
        }
        for region in ["WT", "TC", "ET"]
    }

    try:

        with torch.no_grad():

            for images, masks in tqdm(
                val_loader,
                desc="Validation",
                leave=False
            ):

                images = images.to(
                    device,
                    non_blocking=True
                )

                masks = masks.to(
                    device,
                    non_blocking=True
                ).long()

                with autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda")
                ):

                    outputs = model(images)

                    loss, ce_loss, dice_loss = criterion(
                        outputs,
                        masks
                    )

                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_dice_loss += dice_loss.item()

                num_batches += 1

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                predictions = predictions.cpu().numpy()
                masks_np = masks.cpu().numpy()

                for i in range(predictions.shape[0]):

                    pred_regions = get_regions(
                        predictions[i]
                    )

                    target_regions = get_regions(
                        masks_np[i]
                    )

                    for region in ["WT", "TC", "ET"]:

                        pred_mask = pred_regions[region]
                        target_mask = target_regions[region]

                        metrics = calculate_binary_metrics(
                            pred_mask,
                            target_mask
                        )

                        if metrics is not None:

                            for key in [
                                "dice",
                                "precision",
                                "recall",
                                "iou"
                            ]:

                                region_results[region][key].append(
                                    metrics[key]
                                )

                        hd95 = calculate_hd95(
                            pred_mask,
                            target_mask
                        )

                        if np.isfinite(hd95):

                            region_results[region]["hd95"].append(
                                hd95
                            )

    finally:

        model.train(was_training)

    if num_batches == 0:
        raise RuntimeError("Validation loader is empty.")

    results = {
        "val_loss": total_loss / num_batches,
        "val_ce": total_ce / num_batches,
        "val_dice_loss": total_dice_loss / num_batches
    }

    for region in ["WT", "TC", "ET"]:

        for metric in [
            "dice",
            "precision",
            "recall",
            "iou",
            "hd95"
        ]:

            values = region_results[region][metric]

            if len(values) > 0:
                results[f"{region}_{metric}"] = float(
                    np.mean(values)
                )
            else:
                results[f"{region}_{metric}"] = np.nan

        results[f"{region}_valid_cases"] = len(
            region_results[region]["dice"]
        )

        results[f"{region}_hd95_valid_cases"] = len(
            region_results[region]["hd95"]
        )

    results["mean_dice"] = float(np.nanmean([
        results["WT_dice"],
        results["TC_dice"],
        results["ET_dice"]
    ]))

    results["mean_hd95"] = float(np.nanmean([
        results["WT_hd95"],
        results["TC_hd95"],
        results["ET_hd95"]
    ]))

    return results
