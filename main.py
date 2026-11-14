"""Train and score strategies, and compare them with the frontier of static Bengen strategies.

    python main.py                          # 4% rule vs. the optimised example vs. our strategy
    python main.py strategies/run.py   # only your strategy (vs. the 4% rule)
    python main.py strategies/run.py --no-frontier   # faster, skips the frontier

For every strategy file, main.py creates ``Strategy()``, calls ``fit`` on the train
scenarios (data/train.parquet) and evaluates it on the train and test scenarios (by default
data/validation.parquet). The Bengen 4% rule (strategies/bengen.py) is always included as a
baseline. Strategies are not ranked on a single number: main.py prints the withdrawals
(norm_wd), the ruin probability and J = alpha * norm_wd - (1 - alpha) * ruin_prob for
several alphas.

The frontier: strategies/bengen_optimised.py is trained once for every alpha in
FRONTIER_ALPHAS. Each alpha gives the best static allocation and fixed withdrawal rate
for a different balance between withdrawals and ruin. frontier.png shows these as a curve,
with every strategy as a star. Stars above or to the left of the curve do better than any
static Bengen strategy.

We evaluate submissions with this script, using a hidden set of scenarios as --test:

    python main.py strategies/run.py --test holdout.parquet
"""

import argparse
import contextlib
import dataclasses
import importlib.util
import io
import json
import os
import random

import matplotlib
import numpy as np
import torch

from utils import Config, evaluate, load_scenarios

matplotlib.use("Agg")  # only save figures to files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "strategies", "bengen.py")
EXAMPLE = os.path.join(HERE, "strategies", "bengen_optimised.py")
OURS = os.path.join(HERE, "strategies", "run.py")  # our submission, included when no strategy is named
FRONTIER_ALPHAS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 1.0]
SEED = 0


def load_strategy(path):
    """Import the file at ``path`` and return a new ``Strategy()`` from it."""
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Strategy()


def fit_and_score(path, train, test, cfg):
    """Fit the strategy in ``path`` on the train scenarios and score it on train and test."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    strategy = load_strategy(path)
    strategy.fit(train, cfg)
    return {"train": evaluate(strategy, train, cfg), "test": evaluate(strategy, test, cfg)}


def print_table(results, split, alpha):
    names = list(results)
    print(f"\nScores on the {split} scenarios (J with alpha = {alpha}):")
    print(f"  {'metric':<20s}" + "".join(f"{name:>20s}" for name in names))
    for metric in results[names[0]][split]:
        print(f"  {metric:<20s}" + "".join(f"{results[name][split][metric]:>20.4f}" for name in names))


def print_j_per_alpha(results):
    """J = alpha * norm_wd - (1 - alpha) * ruin_prob on the test scenarios, for several alphas."""
    print("\nJ on the test scenarios for different alphas (higher is better):")
    print(f"  {'alpha':<20s}" + "".join(f"{alpha:>8}" for alpha in FRONTIER_ALPHAS))
    for name, r in results.items():
        m = r["test"]
        scores = [alpha * m["norm_wd"] - (1 - alpha) * m["ruin_prob"] for alpha in FRONTIER_ALPHAS]
        print(f"  {name:<20s}" + "".join(f"{j:>8.3f}" for j in scores))


def plot_frontier(frontier, results):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if frontier:
        points = sorted(frontier, key=lambda p: p["test"]["ruin_prob"])
        ruin = [p["test"]["ruin_prob"] for p in points]
        norm_wd = [p["test"]["norm_wd"] for p in points]
        ax.plot(ruin, norm_wd, "-o", color="grey", label="optimised static Bengen (frontier)")
        for p, x, y in zip(points, ruin, norm_wd):
            ax.annotate(f"alpha={p['alpha']}", (x, y), xytext=(6, -10), textcoords="offset points", fontsize=8)
    for name, r in results.items():
        ax.scatter(r["test"]["ruin_prob"], r["test"]["norm_wd"], marker="*", s=180, zorder=3,
                   label=f"{name} (J = {r['test']['J']:.3f})")
    ax.set_xlabel("ruin probability (test)")
    ax.set_ylabel("normalised withdrawals, norm_wd (test)")
    ax.set_title("Strategies vs. the frontier of static Bengen strategies (better: up and left)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig("frontier.png", dpi=150, bbox_inches="tight")
    print("\nSaved the plot to frontier.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("strategies", nargs="*", default=[EXAMPLE, OURS],
                        help="strategy files to evaluate (default: bengen_optimised and run)")
    parser.add_argument("--train", default=os.path.join(HERE, "data", "train.parquet"), help="train scenarios")
    parser.add_argument("--test", default=os.path.join(HERE, "data", "validation.parquet"), help="test scenarios")
    parser.add_argument("--no-frontier", action="store_true", help="skip the frontier (it takes a while)")
    parser.add_argument("--out", help="also save all results to this JSON file")
    args = parser.parse_args()

    cfg = Config()
    train = load_scenarios(args.train)
    test = load_scenarios(args.test)
    print(f"{train.n_scenarios} train and {test.n_scenarios} test scenarios, "
          f"{train.n_years} years, {train.n_assets} assets")

    results = {}
    for path in [BASELINE] + args.strategies:
        name = os.path.basename(path)[:-3]
        print(f"\n== {name} ==")
        results[name] = fit_and_score(os.path.abspath(path), train, test, cfg)

    print_table(results, "train", cfg.alpha)
    print_table(results, "test", cfg.alpha)
    print_j_per_alpha(results)

    frontier = []
    if not args.no_frontier:
        print("\n== frontier: bengen_optimised trained for every alpha ==")
        for alpha in FRONTIER_ALPHAS:
            with contextlib.redirect_stdout(io.StringIO()):  # hide the training output
                point = fit_and_score(EXAMPLE, train, test, dataclasses.replace(cfg, alpha=alpha))
            print(f"  alpha {alpha}: test ruin_prob {point['test']['ruin_prob']:.4f}, "
                  f"norm_wd {point['test']['norm_wd']:.4f}")
            frontier.append({"alpha": alpha, **point})
    plot_frontier(frontier, results)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": dataclasses.asdict(cfg), "results": results, "frontier": frontier}, f, indent=2)


if __name__ == "__main__":
    main()
