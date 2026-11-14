"""Build the datasets for the neural network: one fixed test set and 50 bootstrapped sets.

    python -B bootstrap/datasets.py

What it does:
  1. Merges the 100 train and 80 validation scenarios (180 in total).
  2. Sets a *test set* aside: 15 random train scenarios and 12 random validation scenarios,
     saved to data/test_set.parquet. This is done only once: if data/split_manifest.csv
     exists, the script reuses the split in it and never reads or rewrites the test set.
  3. Saves the remaining 153 scenarios to data/pool.parquet.
  4. Makes 50 sets from the pool. For every set, the pool is first split into scenarios for
     training (80%) and scenarios for validation (20%), differently for every set. Then both
     parts are bootstrapped separately (see sampler.py), so no original scenario ends up in
     both the train and the validation scenarios of the same set.

Use a set like this:

    from bootstrap.datasets import load_set
    train, validation = load_set(0)   # two Scenarios objects, as from utils.load_scenarios

The test set and the pool are in the same format as data/train.parquet, so they can be read
with ``utils.load_scenarios`` and used as ``--test`` in main.py. The sets themselves are
yearly (returns, inflation and market variables per year): bootstrapped scenarios cannot be
stored month by month, because the bond returns are computed from the yield of the month
before, and that breaks where two blocks meet.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # so that this also runs as a script

from bootstrap.sampler import BlockBootstrap  # noqa: E402
from utils import DTYPE, Scenarios, load_scenarios  # noqa: E402

DATA = os.path.join(ROOT, "data")
MANIFEST = os.path.join(DATA, "split_manifest.csv")  # which scenario went where
TEST_SET = os.path.join(DATA, "test_set.parquet")
POOL = os.path.join(DATA, "pool.parquet")
SETS = os.path.join(DATA, "bootstrap_sets")

SEED = 0
N_TEST = {"train": 15, "validation": 12}  # test scenarios taken from each original file
N_SETS = 50
VALIDATION_FRACTION = 0.2  # share of the pool scenarios that every set uses for validation
N_TRAIN_SCENARIOS = 500  # bootstrapped scenarios per set ...
N_VALIDATION_SCENARIOS = 125  # ... for training and for validation
BLOCK_YEARS = 5


def make_test_split():
    """Choose the test scenarios (once) and write the test set, the pool and the manifest."""
    if os.path.exists(MANIFEST):
        print(f"{os.path.relpath(MANIFEST, ROOT)} exists: keeping the existing test set and pool")
        return pd.read_csv(MANIFEST)
    if os.path.exists(TEST_SET):
        raise SystemExit("data/test_set.parquet exists without a manifest: not overwriting it")

    rng = np.random.default_rng(SEED)
    frames, manifest = [], []
    for origin, n_test in N_TEST.items():
        frame = pd.read_parquet(os.path.join(DATA, f"{origin}.parquet"))
        ids = np.sort(frame["history_id"].unique())
        test_ids = set(rng.choice(ids, size=n_test, replace=False))
        manifest += [{"history_id": i, "origin": origin, "role": "test" if i in test_ids else "pool"} for i in ids]
        frames.append(frame)
    manifest = pd.DataFrame(manifest)
    if manifest["history_id"].duplicated().any():
        raise SystemExit("the train and validation files share scenario ids")

    merged = pd.concat(frames, ignore_index=True)
    role = merged["history_id"].map(manifest.set_index("history_id")["role"])
    merged[role == "test"].to_parquet(TEST_SET, index=False)
    merged[role == "pool"].to_parquet(POOL, index=False)
    manifest.to_csv(MANIFEST, index=False)
    print(f"test set: {(manifest.role == 'test').sum()} scenarios, pool: {(manifest.role == 'pool').sum()} scenarios")
    return manifest


def make_sets(manifest):
    """Write the bootstrapped sets, each from its own train/validation split of the pool."""
    pool = load_scenarios(POOL)  # scenarios are sorted by id
    pool_manifest = manifest[manifest.role == "pool"].sort_values("history_id").reset_index(drop=True)
    if len(pool_manifest) != pool.n_scenarios:
        raise SystemExit("data/pool.parquet does not match the manifest")
    os.makedirs(SETS, exist_ok=True)

    for k in range(N_SETS):
        rng = np.random.default_rng([SEED, k])
        # Split per original file, so every set validates on scenarios of both kinds.
        is_validation = np.zeros(pool.n_scenarios, dtype=bool)
        for origin in N_TEST:
            index = np.flatnonzero(pool_manifest.origin == origin)
            n_validation = round(len(index) * VALIDATION_FRACTION)
            is_validation[rng.choice(index, size=n_validation, replace=False)] = True

        parts = {}
        for name, mask, n, seed in [("train", ~is_validation, N_TRAIN_SCENARIOS, 2 * k),
                                    ("validation", is_validation, N_VALIDATION_SCENARIOS, 2 * k + 1)]:
            source = pool.subset(torch.from_numpy(np.flatnonzero(mask)))
            new = BlockBootstrap(source, BLOCK_YEARS, seed=seed).sample(n)
            parts[name] = {
                "returns": new.returns.float(),  # float32 halves the file size
                "inflation": new.inflation.float(),
                "market": new.market.float(),
                "source_ids": list(pool_manifest.history_id[mask]),
            }
        torch.save({**parts, "asset_names": pool.asset_names, "market_names": pool.market_names,
                    "block_years": BLOCK_YEARS}, set_path(k))
    print(f"{N_SETS} sets in {os.path.relpath(SETS, ROOT)}: {N_TRAIN_SCENARIOS} train and "
          f"{N_VALIDATION_SCENARIOS} validation scenarios each, blocks of {BLOCK_YEARS} years")


def set_path(k):
    return os.path.join(SETS, f"set_{k:02d}.pt")


def load_set(k):
    """Return (train, validation) of set k (0 to 49) as two Scenarios objects."""
    saved = torch.load(set_path(k))

    def scenarios(part):
        return Scenarios(returns=part["returns"].to(DTYPE), inflation=part["inflation"].to(DTYPE),
                         market=part["market"].to(DTYPE), asset_names=saved["asset_names"],
                         market_names=saved["market_names"])

    return scenarios(saved["train"]), scenarios(saved["validation"])


if __name__ == "__main__":
    make_sets(make_test_split())
