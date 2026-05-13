"""Plotting helpers for simulation CSV logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(input_csv: str | Path, out: str | Path) -> None:
    df = pd.read_csv(input_csv)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    plots = [
        ("communicative_success_rate", "Success"),
        ("contradiction_rate", "Contradictions"),
        ("lexical_alignment", "Lexical alignment"),
        ("active_mappings", "Active mappings"),
        ("compression_proxy", "Compression proxy"),
    ]
    for ax, (col, title) in zip(axes.flat, plots):
        if col in df:
            ax.plot(df["episode"], df[col])
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
