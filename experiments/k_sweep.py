"""Quick sweep of the withdrawal factor K, with the portfolio weights kept fixed.

    python -B experiments/k_sweep.py

Fits strategies/run.py once (K = 0.9) and then changes only K, so the effect of the
withdrawal rule is seen on its own. For every K it shows ruin and withdrawals on the train,
validation and test-set scenarios, and how often the limits in Config bind: the 3% minimum
(wd_min) and the 12% maximum (wd_max, a share of the initial capital in today's money).

K <= 0 asks for nothing, so the simulator raises it to the 3% minimum every year.
For retraining the weights for every K, use experiments/bootstrap_sets.py --k ...
"""

import contextlib
import io
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bootstrap.datasets import TEST_SET  # noqa: E402
from strategies.run import Strategy  # noqa: E402
from utils import Config, load_scenarios, simulate  # noqa: E402

K_VALUES = [-0.5, 0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5, 2.0]


def main():
    cfg = Config()
    data = {name: load_scenarios(os.path.join(ROOT, "data", f"{name}.parquet"))
            for name in ["train", "validation"]}
    data["test_set"] = load_scenarios(TEST_SET)

    torch.manual_seed(0)
    strategy = Strategy()
    with contextlib.redirect_stdout(io.StringIO()):
        strategy.fit(data["train"], cfg)

    header = "".join(f"{name + ' ruin/wd':>22s}" for name in data)
    print(f"{'K':>5s}{header}{'at 3% min':>11s}{'at 12% max':>12s}{'final W':>9s}  (last three: validation)")
    for k in K_VALUES:
        strategy.k = k
        cells = ""
        for name, sc in data.items():
            with torch.no_grad():
                out = simulate(strategy, sc, cfg)
            ruin = (out["margins"] <= 0).any(dim=0).double().mean().item()
            cells += f"{ruin:12.3f} / {out['norm_wd'].mean().item():.3f}"
            if name == "validation":
                wd = out["withdrawals"]
                at_min = (wd <= cfg.wd_min + 1e-9).double().mean().item()
                at_max = (wd >= cfg.wd_max - 1e-9).double().mean().item()
                final = out["wealth"][-1].median().item()
        print(f"{k:5.2f}{cells}{at_min:11.1%}{at_max:12.1%}{final:9.2f}")


if __name__ == "__main__":
    main()
