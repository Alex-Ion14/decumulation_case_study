"""Check a strategy on the data team's bootstrap sets, and on their test set.

    python -B bootstrap/datasets.py                       # once: creates data/bootstrap_sets/
    python -B experiments/bootstrap_sets.py               # strategies/run.py, sets 0-9
    python -B experiments/bootstrap_sets.py --sets 50 --k 0.8 0.9 1.0

For every set and every withdrawal factor k it:
  1. fits the strategy on the set's 500 bootstrapped train scenarios,
  2. scores it on the set's 125 bootstrapped validation scenarios,
  3. scores the same fitted strategy on data/test_set.parquet (27 real scenarios that are in
     no set).
It prints the mean and the spread over the sets, so you can see whether a difference between
two choices of k is larger than the noise from the data.

Use it to CHOOSE things (such as k). The official score is still
`python main.py strategies/run.py`. That command does not use these sets, because they also
contain validation scenarios and the README says `fit` may only see the train scenarios.

Warning from the data team: the bootstrap softens the worst scenarios, so the ruin
probabilities here are lower than on scenarios that were never cut up. Compare choices with
each other; don't read the numbers as a forecast.
"""

import argparse
import contextlib
import importlib.util
import io
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bootstrap.datasets import TEST_SET, load_set, set_path  # noqa: E402
from strategies.withdrawal import K  # noqa: E402
from utils import Config, evaluate, load_scenarios  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_strategy_class(path):
    spec = importlib.util.spec_from_file_location("strategy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("strategy", nargs="?", default=os.path.join(ROOT, "strategies", "run.py"))
    parser.add_argument("--sets", type=int, default=10, help="use sets 0 .. SETS-1 (at most 50)")
    parser.add_argument("--k", type=float, nargs="+", default=[K], help="withdrawal factors to compare")
    parser.add_argument("--weights", choices=["bengen", "learned"], default="bengen",
                        help="bengen: fixed 50/50 US equity / 5Y Treasury; learned: 7 weights trained per set")
    args = parser.parse_args()

    if not os.path.exists(set_path(0)):
        raise SystemExit("no bootstrap sets found: run  python -B bootstrap/datasets.py  first")
    Strategy = load_strategy_class(args.strategy)
    cfg = Config()
    test = load_scenarios(TEST_SET)

    results = {k: {"val": [], "test": []} for k in args.k}  # lists of (ruin_prob, norm_wd)
    for s in range(args.sets):
        train, validation = load_set(s)
        for k in args.k:
            torch.manual_seed(s)
            strategy = Strategy(k=k, seed=s, weights=args.weights)
            with contextlib.redirect_stdout(io.StringIO()):  # hide the training output
                strategy.fit(train, cfg)
            for name, scenarios in [("val", validation), ("test", test)]:
                m = evaluate(strategy, scenarios, cfg)
                results[k][name].append((m["ruin_prob"], m["norm_wd"]))
        print(f"  set {s:2d} done", flush=True)

    for name, label in [("val", "bootstrapped validation of each set"), ("test", "test_set.parquet")]:
        print(f"\nOn the {label} (mean +/- std over {args.sets} sets):")
        print(f"  {'k':>5s} {'ruin_prob':>17s} {'norm_wd':>17s}   J at alpha " + " ".join(f"{a:>6}" for a in ALPHAS))
        for k in args.k:
            ruin, wd = np.array(results[k][name]).T
            js = " ".join(f"{(a * wd - (1 - a) * ruin).mean():6.3f}" for a in ALPHAS)
            print(f"  {k:5.2f} {ruin.mean():8.3f} +/- {ruin.std():.3f} {wd.mean():8.3f} +/- {wd.std():.3f}   {js:>s}")


if __name__ == "__main__":
    main()
