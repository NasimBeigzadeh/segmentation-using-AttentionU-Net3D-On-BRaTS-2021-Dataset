
import torch
from tqdm.auto import tqdm
from torch.amp import autocast

from .checkpoint import save_checkpoint
from ..evaluation.validate import validate_model


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scaler,
    device,
    start_epoch,
    target_epochs,
    history,
    best_val_loss,
    best_epoch,
    best_model_state_dict,
    checkpoint_path,
    config,
    max_train_steps=None
):

    for epoch in range(start_epoch, target_epochs + 1):

        print("\n" + "=" * 65)
        print(f"EPOCH {epoch}/{target_epochs}")
        print("=" * 65)

        model.train()

        train_loss = 0.0
        train_ce = 0.0
        train_dice = 0.0

        num_steps = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}"
        )

        for step, (images, masks) in enumerate(progress):

            if (
                max_train_steps is not None
                and step >= max_train_steps
            ):
                break

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            ).long()

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                outputs = model(images)

                loss, ce_loss, dice_loss = criterion(
                    outputs,
                    masks
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_ce += ce_loss.item()
            train_dice += dice_loss.item()

            num_steps += 1

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "CE": f"{ce_loss.item():.4f}",
                "Dice": f"{dice_loss.item():.4f}"
            })

        if num_steps == 0:
            raise RuntimeError("No training steps completed.")

        train_loss /= num_steps
        train_ce /= num_steps
        train_dice /= num_steps

        val_results = validate_model(
            model,
            val_loader,
            criterion,
            device
        )

        val_loss = val_results["val_loss"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_dice_loss": train_dice,
            **val_results,
            "train_steps": num_steps
        }

        history.append(epoch_record)

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            best_model_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            print("\nNew best model found.")

        save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            history=history,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            best_model_state_dict=best_model_state_dict,
            config=config
        )

        print("\nTrain Loss:", train_loss)
        print("Validation Loss:", val_loss)
        print("Mean Dice:", val_results["mean_dice"])
        print("Mean HD95:", val_results["mean_hd95"])
        print("Best Epoch:", best_epoch)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_model_state_dict": best_model_state_dict
    }
