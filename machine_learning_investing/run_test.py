"""The final check: the market-aware MLP and the Bengen rule on the test set.

    python -B machine_learning_investing/run_test.py
    python -B machine_learning_investing/run_test.py --plot-only   # only redraw the figure

THIS READS data/test_set.parquet, the 27 scenarios that were set aside. Every look at the test
set makes it a little less of a clean check, so do not use these results to change the models.

What it does:
  1. Trains the market-aware MLP exactly as in experiment 1: one network per bootstrapped set
     (the first N_MODELS sets), so we see how much the result depends on the set.
  2. Scores every network, their average portfolio (the "ensemble") and the Bengen 4% rule
     on the test set, with the case's own evaluation (utils.evaluate) and with our score.
  3. Draws the case's frontier plot: strategies/bengen_optimised.py trained for every alpha
     and scored on the test set. Unlike main.py's default, the frontier is trained on the
     pool (data/pool.parquet) and not on data/train.parquet, because that file contains 15
     of the test scenarios.

Writes to machine_learning_investing/results/:
  test_strategies.csv  the metrics of every strategy on the test set
  test_scenarios.csv   final wealth and ruin per test scenario, for Bengen and the networks
  test_frontier.csv    the points of the frontier
  test_frontier.png    the frontier plot
"""

import contextlib
import dataclasses
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")  # only save figures to files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so that this also runs as a script

import main as case  # noqa: E402  (the case's main.py: the frontier is computed with its functions)
from bootstrap.datasets import MANIFEST, POOL, TEST_SET  # noqa: E402
from machine_learning_investing.data import load_fold  # noqa: E402
from machine_learning_investing.policies import Ensemble  # noqa: E402
from machine_learning_investing.run_comparison import MODELS, N_PROCESSES, RESULTS  # noqa: E402
from machine_learning_investing.training import metrics, outcomes, train  # noqa: E402
from utils import Config, evaluate, load_scenarios  # noqa: E402

N_MODELS = 20
MODEL = "MLP, with market"
BENGEN, NETWORK, ENSEMBLE = "Bengen 4% rule", "Market MLP (average of the networks)", "Market MLP ensemble"
COLORS = {BENGEN: "#2a78d6", NETWORK: "#eb6834", ENSEMBLE: "#1baf7a"}
TEXT, MUTED, GRID, SURFACE, FRONTIER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"


def train_network(k):
    torch.set_num_threads(1)  # the networks are trained next to each other, one processor each
    fold = load_fold(k)
    make, settings = MODELS[MODEL]
    return train(make(), fold.train, fold.validation, Config(), seed=k, **settings)[0]


def score(strategy, scenarios, cfg):
    """The case's metrics (J, ruin_prob, norm_wd, ...) and ours (score, final wealth, ...)."""
    ours = metrics(strategy, scenarios, cfg)
    return {**evaluate(strategy, scenarios, cfg), "score": ours["score"],
            "mean_final_wealth": ours["mean_final_wealth"], "p10_final_wealth": ours["p10_final_wealth"],
            "growth_share": ours["growth_share"], "turnover": ours["turnover"]}


def frontier_points(pool, test, cfg):
    points = []
    for alpha in case.FRONTIER_ALPHAS:
        with contextlib.redirect_stdout(io.StringIO()):  # hide the training output
            point = case.fit_and_score(case.EXAMPLE, pool, test, dataclasses.replace(cfg, alpha=alpha))
        points.append({"alpha": alpha, **point["test"]})
    return pd.DataFrame(points)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))


def _draw_strategies(ax, strategies, star_size, labels):
    """The single networks as dots and the three named strategies as stars."""
    single = strategies[strategies.strategy.str.startswith("network ")]
    ax.scatter(single.ruin_prob, single.norm_wd, s=star_size / 9, color=COLORS[NETWORK], alpha=0.35, linewidth=0,
               zorder=3, label=f"single market MLPs ({len(single)})")
    offsets = {BENGEN: (14, 16, "left"), NETWORK: (0, -26, "center"), ENSEMBLE: (0, 18, "center")}
    for name in (BENGEN, NETWORK, ENSEMBLE):
        row = strategies[strategies.strategy == name].iloc[0]
        ax.scatter(row.ruin_prob, row.norm_wd, marker="*", s=star_size, color=COLORS[name], edgecolor=SURFACE,
                   linewidth=1.2, zorder=4, label=f"{name}: ruin {row.ruin_prob:.1%}, norm_wd {row.norm_wd:.3f}")
        if labels:
            x, y, align = offsets[name]
            ax.annotate(name, (row.ruin_prob, row.norm_wd), xytext=(x, y), textcoords="offset points",
                        fontsize=8.5, color=TEXT, ha=align, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})


def plot_frontier(frontier, strategies, n_test):
    fig, ax = plt.subplots(figsize=(10.5, 6.6), facecolor=SURFACE)
    curve = frontier.sort_values("ruin_prob")
    ax.plot(curve.ruin_prob, curve.norm_wd, "-o", color=FRONTIER, linewidth=2, markersize=5, zorder=2,
            label="optimised static Bengen (frontier)")
    for point in curve.itertuples():
        ax.annotate(f"alpha={point.alpha:g}", (point.ruin_prob, point.norm_wd), xytext=(-7, 7),
                    textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    _draw_strategies(ax, strategies, star_size=260, labels=False)
    ax.set_xlim(left=0)
    ax.set_xlabel("ruin probability (test set)", color=MUTED, fontsize=10)
    ax.set_ylabel("normalised withdrawals, norm_wd (test set)", color=MUTED, fontsize=10)
    _style(ax)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False, labelcolor=MUTED)

    # The strategies with a fixed 4% withdrawal lie almost on top of each other: zoom in on them.
    fixed = strategies[strategies.strategy != ""]
    zoom = ax.inset_axes([0.5, 0.07, 0.47, 0.4])
    zoom.plot(curve.ruin_prob, curve.norm_wd, "-o", color=FRONTIER, linewidth=2, markersize=5, zorder=2)
    _draw_strategies(zoom, strategies, star_size=330, labels=True)
    zoom.set_xlim(fixed.ruin_prob.min() - 0.035, fixed.ruin_prob.max() + 0.035)
    zoom.set_ylim(fixed.norm_wd.min() - 0.012, fixed.norm_wd.max() + 0.012)
    zoom.set_title("zoom on the fixed 4% strategies", loc="left", fontsize=9, color=MUTED)
    _style(zoom)
    for side in zoom.spines.values():
        side.set_color(GRID)
    ax.indicate_inset_zoom(zoom, edgecolor=MUTED, alpha=0.6)

    fig.suptitle("Market MLP and Bengen against the frontier of static strategies", x=0.02, ha="left",
                 fontsize=14, color=TEXT)
    fig.text(0.02, 0.915, f"{n_test} test scenarios. Better: up and to the left. Bengen and the MLP both withdraw "
                          "a fixed 4%, so they can only differ in ruin (left-right).",
             ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = os.path.join(RESULTS, "test_frontier.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {os.path.relpath(path)}")


def run():
    os.makedirs(RESULTS, exist_ok=True)
    cfg = Config()
    with ProcessPoolExecutor(N_PROCESSES) as pool_of_processes:
        networks = list(pool_of_processes.map(train_network, range(N_MODELS)))

    test, pool = load_scenarios(TEST_SET), load_scenarios(POOL)
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest.role == "test"].sort_values("history_id")  # the order of load_scenarios
    assert len(manifest) == test.n_scenarios

    candidates = {BENGEN: case.load_strategy(case.BASELINE), ENSEMBLE: Ensemble(networks),
                  **{f"network {k:02d}": network for k, network in enumerate(networks)}}
    rows, scenario_rows = [], []
    for name, strategy in candidates.items():
        rows.append({"strategy": name, **score(strategy, test, cfg)})
        out = outcomes(strategy, test, cfg)
        for i, (history_id, origin) in enumerate(zip(manifest.history_id, manifest.origin)):
            scenario_rows.append({"strategy": name, "history_id": history_id, "origin": origin,
                                  "final_wealth": out["final_wealth"][i].item(), "ruined": bool(out["ruined"][i])})
    strategies = pd.DataFrame(rows)
    single = strategies[strategies.strategy.str.startswith("network ")]
    average = {"strategy": NETWORK, **single.drop(columns="strategy").mean().to_dict()}
    strategies = pd.concat([strategies.iloc[:2], pd.DataFrame([average]), strategies.iloc[2:]], ignore_index=True)
    scenarios = pd.DataFrame(scenario_rows)
    frontier = frontier_points(pool, test, cfg)

    strategies.to_csv(os.path.join(RESULTS, "test_strategies.csv"), index=False)
    scenarios.to_csv(os.path.join(RESULTS, "test_scenarios.csv"), index=False)
    frontier.to_csv(os.path.join(RESULTS, "test_frontier.csv"), index=False)

    pd.set_option("display.width", 250)
    columns = ["strategy", "ruin_prob", "norm_wd", "J", "median_final_wealth", "mean_final_wealth",
               "p10_final_wealth", "score", "turnover"]
    print(f"Test set: {test.n_scenarios} scenarios")
    print(strategies[columns].head(3).round(3).to_string(index=False))
    print(f"\nThe {len(single)} single networks:")
    print(single[columns[1:]].describe().loc[["mean", "std", "min", "50%", "max"]].round(3).to_string())
    print("\nFrontier (optimised static Bengen, trained on the pool):")
    print(frontier[["alpha", "ruin_prob", "norm_wd", "median_final_wealth"]].round(3).to_string(index=False))

    wide = scenarios.pivot(index=["history_id", "origin"], columns="strategy", values="final_wealth").reset_index()
    network_columns = [c for c in wide.columns if c.startswith("network ")]
    wide["networks"] = wide[network_columns].mean(axis=1)
    print("\nPer test scenario: final wealth, Bengen against the average of the single networks")
    print(f"  the networks end with more wealth in {(wide.networks > wide[BENGEN]).sum()} of {len(wide)} scenarios")
    ruined = scenarios.assign(kind=scenarios.strategy.where(~scenarios.strategy.str.startswith("network "), "networks"))
    print(ruined.groupby(["kind", "origin"]).ruined.mean().unstack().round(3).to_string())
    plot_frontier(frontier, strategies, test.n_scenarios)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw the figure from the saved tables, without training again
        saved = [pd.read_csv(os.path.join(RESULTS, f"test_{name}.csv")) for name in ("frontier", "strategies", "scenarios")]
        plot_frontier(saved[0], saved[1], saved[2].history_id.nunique())
    else:
        run()
