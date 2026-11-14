"""Figures for the two experiments (run run_comparison.py and run_confidence.py first).

    python -B machine_learning_investing/plots.py

Saves in machine_learning_investing/results/:
  comparison_models.png    held-out score, ruin probability and median final wealth per model
  allocation_all_models.png  the average portfolio per year of every model in experiment 1
  allocation_by_year.png   the average portfolio per year of the two "confident" models
  confidence_by_year.png   how much the networks agree about the portfolio, year by year
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # only save figures to files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BLUE, BLUE_LIGHT, BLUE_DARK = "#2a78d6", "#9ec5f4", "#0d366b"
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# One fixed colour per asset, in the order of utils.ASSET_NAMES.
ASSET_COLORS = {"US equity": "#2a78d6", "EM equity": "#eb6834", "US REIT": "#1baf7a", "US Treasury 5Y": "#eda100",
                "US Treasury 30Y": "#e87ba4", "US TIPS 10Y": "#008300", "Cash": "#6250d6"}
percent = FuncFormatter(lambda v, _: f"{v:.0%}")


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {os.path.relpath(path)}")


def comparison_models():
    folds = pd.read_csv(os.path.join(RESULTS, "comparison_folds.csv"))
    held = folds[folds.part == "held_out"]
    models = list(held.model.unique())[::-1]  # first model at the top
    # metric, title, format of the axis, format of the label on the average
    panels = [("score", "Score (higher is better)", "{:.1f}", "{:.2f}"),
              ("ruin_prob", "Ruin probability (lower is better)", "{:.0%}", "{:.1%}"),
              ("median_final_wealth", "Median final wealth (start = 1)", "{:.0f}", "{:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True, facecolor=SURFACE)
    for ax, (metric, title, axis_format, label_format) in zip(axes, panels):
        for y, model in enumerate(models):
            values = held[held.model == model][metric].to_numpy()
            ax.scatter(values, np.full(len(values), y), s=14, color=BLUE_LIGHT, alpha=0.8, linewidth=0, zorder=2)
            ax.scatter(values.mean(), y, s=90, color=BLUE_DARK, zorder=3, edgecolor=SURFACE, linewidth=2)
            ax.annotate(label_format.format(values.mean()), (values.mean(), y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=9, color=TEXT)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, f=axis_format: f.format(v)))
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(-0.6, len(models) - 0.3)
        style(ax)
    for label in axes[0].get_yticklabels():
        label.set_color(TEXT)
        label.set_fontsize(10)
    n_folds = held.fold.nunique()
    fig.suptitle("Investing models on scenarios they have never seen", x=0.012, ha="left", fontsize=14, color=TEXT)
    fig.text(0.012, 0.915, f"Large dot: average over {n_folds} folds. Small dots: the single folds "
                           "(31 original held-out scenarios each). Fixed 4% withdrawal.",
             ha="left", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "comparison_models.png")


def allocation_figure(averages, title, subtitle, file_name, n_columns):
    """One stacked-area panel per model. averages: {model name: table with a row per year and a column per asset}."""
    assets = list(ASSET_COLORS)
    n_rows = -(-len(averages) // n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(5.4 * n_columns, 4.4 * n_rows + 1.2), sharex=True,
                             sharey=True, facecolor=SURFACE, squeeze=False)
    for ax, (name, average) in zip(axes.flat, averages.items()):
        ax.stackplot(average.index, average[assets].to_numpy().T, colors=[ASSET_COLORS[a] for a in assets],
                     edgecolor=SURFACE, linewidth=1.2)
        # label every layer in the middle of the plot (if it is thick enough there to hold a label)
        middle, bottom = len(average) // 2, 0.0
        for asset in assets:
            share = average[asset].iloc[middle]
            if share > 0.06:
                ax.text(average.index[middle], bottom + share / 2, f"{asset} {share:.0%}", ha="center",
                        va="center", fontsize=9, color="white", fontweight="bold")
            bottom += share
        ax.set_title(name, loc="left", fontsize=11, color=TEXT)
        ax.set_xlim(1, 30)
        ax.set_ylim(0, 1)
        ax.set_xticks([1, 10, 20, 30])
        ax.yaxis.set_major_formatter(percent)
        style(ax, "y")
    for ax in axes.flat[len(averages):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("year", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of the portfolio", color=MUTED, fontsize=9)
    top = 1 - 0.75 / fig.get_figheight()  # the title block is the same height in every figure
    handles = [plt.Rectangle((0, 0), 1, 1, color=ASSET_COLORS[a]) for a in assets]
    fig.legend(handles, assets, loc="upper right", bbox_to_anchor=(0.995, top + 0.012), ncol=7, frameon=False,
               fontsize=9, labelcolor=MUTED, handlelength=1.2, columnspacing=1.2)
    fig.suptitle(title, x=0.012, y=0.995, ha="left", fontsize=14, color=TEXT)
    fig.text(0.012, top + 0.035, subtitle, ha="left", va="top", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, top - 0.02))
    save(fig, file_name)


def allocation_all_models():
    """The average portfolio per year of every model in experiment 1."""
    allocations = pd.read_csv(os.path.join(RESULTS, "comparison_allocations.csv"))
    assets = list(ASSET_COLORS)
    averages = {}
    for model, rows in allocations.groupby("model", sort=False):
        # average over the folds, counting every fold by its number of scenarios with money
        weighted = rows[assets].mul(rows.scenarios, axis=0).groupby(rows.year).sum()
        averages[model] = weighted.div(rows.groupby("year").scenarios.sum(), axis=0)
    allocation_figure(
        averages, "Average portfolio per year of every model, on held-out scenarios",
        f"Average over {allocations.fold.nunique()} folds and over the scenarios that still have money in that year.",
        "allocation_all_models.png", n_columns=4)


def allocation_by_year(decisions):
    """The average portfolio per year of the two models with a confidence (experiment 2)."""
    averages = {method: rows[rows.wealth > 0].groupby("year")[list(ASSET_COLORS)].mean()
                for method, rows in decisions.groupby("method", sort=False)}
    allocation_figure(averages, "Average portfolio per year on held-out scenarios",
                      "Average over the scenarios that still have money in that year.",
                      "allocation_by_year.png", n_columns=2)


def confidence_by_year(decisions):
    methods = list(decisions.method.unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(7.2 * len(methods), 4.8), sharey=True, facecolor=SURFACE)
    for ax, method in zip(np.atleast_1d(axes), methods):
        alive = decisions[(decisions.method == method) & (decisions.wealth > 0)]
        by_year = alive.groupby("year").agreement.quantile([0.1, 0.5, 0.9]).unstack()
        ax.fill_between(by_year.index, by_year[0.1], by_year[0.9], color=BLUE_LIGHT, alpha=0.55, linewidth=0)
        ax.plot(by_year.index, by_year[0.5], color=BLUE_DARK, linewidth=2)
        ax.annotate("median", (by_year.index[-1], by_year[0.5].iloc[-1]), xytext=(-6, 8),
                    textcoords="offset points", ha="right", fontsize=9, color=TEXT)
        ax.annotate("middle 80% of scenarios", (by_year.index[14], by_year[0.1].iloc[14]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=9, color=MUTED)
        ax.set_title(method, loc="left", fontsize=11, color=TEXT)
        ax.set_xlim(1, 30)
        ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("year", color=MUTED, fontsize=9)
        ax.yaxis.set_major_formatter(percent)
        style(ax, "y")
    np.atleast_1d(axes)[0].set_ylabel("agreement between the networks", color=MUTED, fontsize=9)
    fig.suptitle("How sure are the models about the portfolio?", x=0.012, ha="left", fontsize=14, color=TEXT)
    fig.text(0.012, 0.9, "Agreement: the share of the portfolio that a network has in common with the average "
                         "portfolio (100% = all networks propose the same).", ha="left", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    save(fig, "confidence_by_year.png")


if __name__ == "__main__":
    comparison_models()
    if os.path.exists(os.path.join(RESULTS, "comparison_allocations.csv")):
        allocation_all_models()
    decisions_path = os.path.join(RESULTS, "confidence_decisions.csv")
    if os.path.exists(decisions_path):
        decisions = pd.read_csv(decisions_path)
        allocation_by_year(decisions)
        confidence_by_year(decisions)
