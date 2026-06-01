from __future__ import annotations

import csv
from pathlib import Path


def make_early_stopping(patience: int, monitor: str = "val/loss"):
    from lightning.pytorch.callbacks import EarlyStopping
    return EarlyStopping(monitor=monitor, patience=patience, mode="min", verbose=True)


def plot_training_curves(output_dir: Path) -> None:
    """Read CSVLogger metrics.csv and write training_curves.png to output_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs_dir = output_dir / "logs"
    if not logs_dir.exists():
        print("  no logs/ dir found — skipping training curves")
        return
    candidates = sorted(logs_dir.glob("version_*/metrics.csv"))
    if not candidates:
        print("  no metrics.csv found — skipping training curves")
        return
    csv_path = candidates[-1]

    rows: list[dict[str, float | None]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | None] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = None
            rows.append(parsed)

    epoch_rows = [r for r in rows if r.get("val/loss") is not None]
    if not epoch_rows:
        print("  no epoch-level rows found — skipping training curves")
        return

    metrics = [
        ("loss",        "Total Loss"),
        ("onset_loss",  "Onset Loss"),
        ("offset_loss", "Offset Loss"),
        ("active_loss", "Active Loss"),
        ("onset_peak",  "Onset Peak Prob"),
        ("offset_peak", "Offset Peak Prob"),
        ("active_mean", "Active Mean Prob"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)), sharex=True)
    for ax, (key, title) in zip(axes, metrics):
        t_key = f"train/{key}_epoch"
        v_key = f"val/{key}"
        t_epochs = [r["epoch"] for r in epoch_rows if r.get(t_key) is not None]
        t_vals   = [r[t_key]   for r in epoch_rows if r.get(t_key) is not None]
        v_epochs = [r["epoch"] for r in epoch_rows if r.get(v_key) is not None]
        v_vals   = [r[v_key]   for r in epoch_rows if r.get(v_key) is not None]
        if t_vals:
            ax.plot(t_epochs, t_vals, label="train", color="#2980b9", linewidth=1.2)
        if v_vals:
            ax.plot(v_epochs, v_vals, label="val", color="#e74c3c", linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylabel("Value", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    out = output_dir / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"training curves -> {out}")
