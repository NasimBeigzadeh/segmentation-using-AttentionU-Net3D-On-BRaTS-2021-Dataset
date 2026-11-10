
import os
import random
import numpy as np
import torch


def get_rng_state():

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state()
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path, model, optimizer, scaler, device):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    required_keys = [
        "model_state_dict",
        "optimizer_state_dict",
        "history",
        "last_epoch",
        "best_val_loss",
        "best_epoch",
        "best_model_state_dict"
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(
                f"Missing checkpoint key: {key}"
            )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    if checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    if checkpoint.get("rng_state") is not None:
        restore_rng_state(checkpoint["rng_state"])

    return checkpoint


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scaler,
    history,
    best_val_loss,
    best_epoch,
    best_model_state_dict,
    config
):

    path = os.path.abspath(path)
    temp_path = path + ".tmp"

    save_data = {

        "last_epoch": epoch,

        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },

        "optimizer_state_dict": optimizer.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,

        "best_model_state_dict": best_model_state_dict,

        "history": history,
        "config": config,

        "rng_state": get_rng_state()
    }

    try:

        torch.save(save_data, temp_path)

        if not os.path.exists(temp_path):
            raise RuntimeError("Checkpoint was not created.")

        if os.path.getsize(temp_path) == 0:
            raise RuntimeError("Checkpoint file is empty.")

        os.replace(temp_path, path)

    except Exception:

        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise

    print("Checkpoint saved:", path)
