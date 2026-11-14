"""The data for the investing models: the bootstrapped sets, plus the original held-out scenarios.

Every bootstrapped set (see bootstrap/datasets.py) splits the 153 pool scenarios into 122 that
are behind its train scenarios and 31 that are behind its validation scenarios. A *fold* here
is one set with three parts:

  train       500 bootstrapped scenarios: the model is trained on these
  validation  125 bootstrapped scenarios: used to decide when to stop training
  held_out    the 31 original (not glued) scenarios behind the validation part: the model has
              never seen any year of them, so they give the fairest score we can compute
              without touching the test set

The test set (data/test_set.parquet) is never read here.
"""

import functools
import os
from dataclasses import dataclass

import pandas as pd
import torch

from bootstrap.datasets import MANIFEST, POOL, load_set, set_path
from utils import Scenarios, load_scenarios


@dataclass
class Fold:
    train: Scenarios
    validation: Scenarios
    held_out: Scenarios
    held_out_ids: list  # the history_id of every held-out scenario
    held_out_origin: list  # "train" or "validation": the original file of every held-out scenario


@functools.lru_cache(maxsize=1)
def _pool():
    """The 153 original pool scenarios with their ids and original file (sorted by id)."""
    if not os.path.exists(set_path(0)):
        raise SystemExit("data/bootstrap_sets is missing: run  python -B bootstrap/datasets.py  first")
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest.role == "pool"].sort_values("history_id").reset_index(drop=True)
    return load_scenarios(POOL), list(manifest.history_id), list(manifest.origin)


def load_fold(k):
    train, validation = load_set(k)
    pool, ids, origins = _pool()
    held_out_ids = set(torch.load(set_path(k))["validation"]["source_ids"])
    index = [i for i, history_id in enumerate(ids) if history_id in held_out_ids]
    return Fold(train, validation, pool.subset(torch.tensor(index)),
                [ids[i] for i in index], [origins[i] for i in index])
