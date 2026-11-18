"""Plot mean training/validation MSE for initialization seeds 1000--1009."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


SEEDS = tuple(range(1000, 1010))


def read_history(output_dir):
    output_dir = Path(output_dir)
    history = np.atleast_1d(np.genfromtxt(
        output_dir / "training_history.csv", delimiter=",", names=True
    ))
    required = {"epoch", "loss_data", "val_loss"}
    if not required.issubset(history.dtype.names or ()) or history.size == 0:
        raise ValueError(f"Missing epoch/training/validation MSE data in {output_dir}")
    epochs = history["epoch"]
    for column in ("epoch", "loss_data", "val_loss"):
        if not np.all(np.isfinite(history[column])) or np.any(history[column] <= 0):
            raise ValueError(f"Expected finite, positive values in {column}.")
    if not np.array_equal(epochs, np.arange(1, len(epochs) + 1)):
        raise ValueError(f"Expected consecutive epochs starting at 1 in {output_dir}")
    return history


def draw_curves(epochs, training_mse, validation_mse, output_dir,
                filename, averaged=False):

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "font.weight": "bold",
        "axes.labelweight": "bold", "axes.linewidth": 0.6,
        "mathtext.fontset": "stix", "svg.fonttype": "none",
    })
    # Match the reference's aspect ratio, margins, grey background and log grid.
    fig, ax = plt.subplots(figsize=(6.44, 4.70))
    fig.subplots_adjust(left=0.108, right=0.920, bottom=0.119, top=0.950)
    ax.set_facecolor("#f5f5f5")
    prefix = "Mean " if averaged else ""
    ax.semilogy(epochs, training_mse, color="#0072bd", linewidth=1.1,
                label=prefix + "Training loss (MSE)")
    ax.semilogy(epochs, validation_mse, color="#d95319", linewidth=1.1,
                linestyle="--", label=prefix + "Validation loss (MSE)")
    ax.set(xlabel="Epoch", ylabel="MSE", xlim=(0, epochs[-1]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, prune="lower"))
    ax.grid(True, which="both", color="#dddddd", linewidth=0.5, alpha=0.8)
    ax.tick_params(which="both", width=0.5, length=2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, prop={"size": 9, "weight": "bold"})
    for extension in ("png", "svg"):
        destination = Path(output_dir) / f"{filename}.{extension}"
        fig.savefig(destination, dpi=1200, facecolor="white")
        print(destination)
    plt.close(fig)


def plot_single_run(output_dir):
    """Preserve per-run plots called by the training script."""
    history = read_history(output_dir)
    draw_curves(history["epoch"], history["loss_data"], history["val_loss"],
                output_dir, "training_validation_loss")


def main(output_dir=None):
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(__file__).resolve().parent / "out_pcnn_5D")
    missing = [output_dir / f"out_{seed}" / "training_history.csv"
               for seed in SEEDS
               if not (output_dir / f"out_{seed}" / "training_history.csv").is_file()]
    if missing:
        raise FileNotFoundError("All ten histories are required. Missing:\n" +
                                "\n".join(map(str, missing)))
    histories = [read_history(output_dir / f"out_{seed}") for seed in SEEDS]
    common_epochs = min(len(history) for history in histories)
    epochs = histories[0]["epoch"][:common_epochs]
    # Average the original MSE values, not their logarithms or penalized loss.
    training_mean = np.mean(np.stack([
        history["loss_data"][:common_epochs] for history in histories
    ]), axis=0)
    validation_mean = np.mean(np.stack([
        history["val_loss"][:common_epochs] for history in histories
    ]), axis=0)
    destination = output_dir / "mean_training_history.csv"
    np.savetxt(destination, np.column_stack([
        epochs, training_mean, validation_mean,
        np.full(common_epochs, len(SEEDS)),
    ]), delimiter=",", header="epoch,mean_training_mse,mean_validation_mse,run_count",
        comments="", fmt=["%d", "%.12g", "%.12g", "%d"])
    draw_curves(epochs, training_mean, validation_mean, output_dir,
                "mean_training_validation_loss", averaged=True)
    print(destination)
    print("Epoch counts: " + ", ".join(
        f"{seed}: {len(history)}" for seed, history in zip(SEEDS, histories)))
    print(f"Mean over {len(SEEDS)} runs; common epochs: {common_epochs}; "
          f"training MSE: {training_mean[-1]:.8f}; "
          f"validation MSE: {validation_mean[-1]:.8f}")


if __name__ == "__main__":
    main()
