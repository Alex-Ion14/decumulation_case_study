"""Path plots of the 7 assets and inflation over the 30 years, one line per scenario.

    python -B plotting/plot_paths.py                       # the 100 train scenarios
    python -B plotting/plot_paths.py data/pool.parquet     # any other scenario file

Saves two figures in the folder plots/ (one panel per asset, plus one for inflation):

  paths_growth_<file>.png          the value of 1 invested at the start, year by year
  paths_yearly_returns_<file>.png  the return in every single year

In every panel a thin line is one scenario, the shaded band holds the middle 90% of the
scenarios and the dark line is the median. Asset returns are *real* (corrected for
inflation), as in the simulation; the inflation panel shows inflation itself.
"""

import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # only save figures to files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # so that this also runs as a script

from utils import load_scenarios  # noqa: E402

OUT = os.path.join(ROOT, "plots")
PATH_COLOR, BAND_COLOR, MEDIAN_COLOR = "#2a78d6", "#9ec5f4", "#0d366b"
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_series(path):
    """Yearly real asset returns and inflation as {panel title: (T, N) array}."""
    sc = load_scenarios(path)
    real = (1 + sc.returns) / (1 + sc.inflation[:, :, None]) - 1
    series = {name: real[:, :, i].numpy() for i, name in enumerate(sc.asset_names)}
    series["Inflation"] = sc.inflation.numpy()
    return series, sc.n_scenarios


def draw_panels(series, transform, title, subtitle, y_label, y_format, log_scale, file_name):
    """One panel per series: every scenario as a thin line, plus the middle 90% and the median."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (name, yearly) in zip(axes.flat, series.items()):
        values = transform(yearly)  # (years on the x-axis, N)
        years = np.arange(len(values)) + (0 if log_scale else 1)  # growth starts at year 0
        low, median, high = np.percentile(values, [5, 50, 95], axis=1)
        ax.fill_between(years, low, high, color=BAND_COLOR, alpha=0.55, linewidth=0, zorder=1)
        ax.plot(years, values, color=PATH_COLOR, alpha=0.22, linewidth=0.6, zorder=2)
        ax.plot(years, median, color=MEDIAN_COLOR, linewidth=2, zorder=3)
        ax.axhline(1 if log_scale else 0, color=MUTED, linewidth=0.8, zorder=0)

        ax.set_title(name, loc="left", fontsize=11, color=TEXT)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        if log_scale:
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(y_format))
        ax.tick_params(which="minor", length=0)
        ax.set_xlim(years[0], years[-1])
        ax.set_xticks([1, 10, 20, 30] if not log_scale else [0, 10, 20, 30])

    for ax in axes[-1]:
        ax.set_xlabel("year", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label, color=MUTED, fontsize=9)

    fig.suptitle(title, x=0.045, y=0.985, ha="left", fontsize=14, color=TEXT)
    fig.text(0.045, 0.94, subtitle, ha="left", fontsize=10, color=MUTED)
    fig.legend(
        handles=[Line2D([], [], color=PATH_COLOR, alpha=0.6, linewidth=0.8, label="one scenario"),
                 Patch(color=BAND_COLOR, alpha=0.55, label="middle 90% of scenarios"),
                 Line2D([], [], color=MEDIAN_COLOR, linewidth=2, label="median")],
        loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=3, frameon=False, fontsize=10, labelcolor=MUTED)
    fig.tight_layout(rect=(0.03, 0, 1, 0.92))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, file_name)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {os.path.relpath(path, ROOT)}")


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "train.parquet")
    label = os.path.splitext(os.path.basename(data))[0]
    series, n = load_series(data)

    def growth(yearly):  # value of 1 at the start, with a first row for year 0
        return np.concatenate([np.ones((1, yearly.shape[1])), np.cumprod(1 + yearly, axis=0)])

    draw_panels(
        series, growth,
        title=f"Growth of 1 invested at the start: {n} scenarios ({label})",
        subtitle="Assets after inflation; the inflation panel shows the price level. "
                 "Same logarithmic scale in every panel.",
        y_label="value (start = 1)", y_format=lambda v, _: f"{v:g}", log_scale=True,
        file_name=f"paths_growth_{label}.png")

    draw_panels(
        series, lambda yearly: yearly,
        title=f"Return in every single year: {n} scenarios ({label})",
        subtitle="Assets after inflation; the inflation panel shows yearly inflation. Same scale in every panel.",
        y_label="yearly return", y_format=lambda v, _: f"{v:.0%}", log_scale=False,
        file_name=f"paths_yearly_returns_{label}.png")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
