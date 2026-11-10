
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(history_df, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training and Validation Loss
    plt.figure(figsize=(10, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        label="Train Loss"
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Attention U-Net - Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Region Dice
    plt.figure(figsize=(10, 5))

    for region in ["WT", "TC", "ET"]:

        plt.plot(
            history_df["epoch"],
            history_df[f"{region}_dice"],
            marker="o",
            label=f"{region} Dice"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Attention U-Net - Region Dice")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir / "region_dice.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Region HD95
    plt.figure(figsize=(10, 5))

    for region in ["WT", "TC", "ET"]:

        plt.plot(
            history_df["epoch"],
            history_df[f"{region}_hd95"],
            marker="o",
            label=f"{region} HD95"
        )

    plt.xlabel("Epoch")
    plt.ylabel("HD95")
    plt.title("Attention U-Net - Region HD95")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir / "region_hd95.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Training plots saved to:", output_dir)
