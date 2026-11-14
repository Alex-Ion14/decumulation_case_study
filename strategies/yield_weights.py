"""The weights team's yield model, trained inside `fit` on the train scenarios only.

The model and its training are those of yield_mlp/yield_mlp.py (the weights team). That file
cannot be imported any more (it loads its data from a folder that was deleted), and it trains
on bootstrap sets made from train AND validation scenarios, which the case rules do not
allow in `fit`. This file keeps their model, score and training loop, and only changes where
the data comes from:

    for every one of N_SETS sets (default 50):
        split the train scenarios that fit receives into 80% and 20% (a different split per set)
        bootstrap 500 scenarios from the 80% and 125 from the 20% (blocks of 5 years)
        train one network on the 500, keep the version that does best on the 125
    invest in the average of the portfolios of all networks

As in yield_mlp.py, the networks are trained with a FIXED 4% withdrawal: the withdrawal rule
of strategies/withdrawal.py plays no part in training the weights.

Speed: training 50 networks one after another takes about 15 seconds each, too slow for the
15-minute limit together with the rest of main.py. So all networks are trained at the same
time: every step simulates the batches of all networks in one go. This gives exactly the same
kind of networks as training them one by one (each has its own numbers, its own batches and
its own best version), only faster.
"""

import torch

from strategies.block_bootstrap import BlockBootstrap, split_scenarios
from utils import DTYPE, Config, Scenarios, simulate

# ── Settings (as in yield_mlp/yield_mlp.py, unless noted) ──────────────────────────────────
YIELDS = ["US_ZC_5Y", "US_ZC_30Y"]  # the two inputs: last year's 5Y and 30Y Treasury yield, in %
WITHDRAWAL_RATE = 0.04  # fixed withdrawal while training the weights
N_SETS = 50  # networks, one per bootstrap set (yield_mlp.py: NUMBER_OF_DATA_SETS)
N_TRAIN, N_VALIDATION = 500, 125  # bootstrapped scenarios per set (as in bootstrap/datasets.py)
VALIDATION_FRACTION = 0.2  # share of the train scenarios kept apart for each set's validation
BLOCK_YEARS = 5
HIDDEN_SIZE = 32
TRAINING_STEPS = 300
SCENARIOS_PER_STEP = 128
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
CHECK_EVERY = 20
RUIN_PENALTY = 1.0
SOFTNESS_START, SOFTNESS_END = 0.1, 0.01


def yields_of(market, market_names):
    """(..., 2) the two yields, in % per year, from market values (..., V)."""
    return torch.stack([market[..., market_names.index(name)] for name in YIELDS], dim=-1)


class Networks:
    """M small networks side by side: 2 yields in, 7 weights out, each with its own numbers.

    Network m is the same as YieldModel in yield_mlp.py: Linear(2, 32) - tanh - Linear(32, 32) -
    tanh - Linear(32, 7) - softmax, with the inputs scaled by the typical yield level.
    """

    def __init__(self, n_models, n_assets, typical, variation, seed=0):
        generator = torch.Generator().manual_seed(seed)
        sizes = [len(YIELDS), HIDDEN_SIZE, HIDDEN_SIZE, n_assets]
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            bound = 1 / n_in ** 0.5  # the default start of torch.nn.Linear
            w = (torch.rand(n_models, n_in, n_out, generator=generator, dtype=DTYPE) * 2 - 1) * bound
            b = (torch.rand(n_models, n_out, generator=generator, dtype=DTYPE) * 2 - 1) * bound
            if i == len(sizes) - 2:  # start close to equal weights, as in yield_mlp.py
                w, b = w * 0.1, torch.zeros_like(b)
            self.layers.append((w.requires_grad_(), b.requires_grad_()))
        self.typical = typical  # (M, 2)
        self.variation = variation  # (M, 2)

    def parameters(self):
        return [p for layer in self.layers for p in layer]

    def portfolios(self, yields):
        """yields (M, S, 2) -> weights (M, S, A): network m gives the portfolios of group m."""
        x = (yields - self.typical[:, None]) / self.variation[:, None]
        for i, (w, b) in enumerate(self.layers):
            x = torch.bmm(x, w) + b[:, None]
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return torch.softmax(x, dim=-1)

    def first_year_portfolios(self):
        """(M, A): the portfolio of every network for typical yields (no yield known in year 0)."""
        return self.portfolios(self.typical[:, None])[:, 0]

    def snapshot(self):
        return [(w.detach().clone(), b.detach().clone()) for w, b in self.layers]


class _Training:
    """Plays every network on its own group of scenarios, with the fixed 4% withdrawal."""

    def __init__(self, networks, n_models):
        self.networks, self.n_models = networks, n_models

    def act(self, obs):
        n = obs.wealth.shape[0]  # M groups of n // M scenarios, group m belongs to network m
        if obs.year == 0:
            w = self.networks.first_year_portfolios()[:, None].expand(-1, n // self.n_models, -1)
        else:
            y = yields_of(obs.market_history[-1], obs.market_names).reshape(self.n_models, -1, len(YIELDS))
            w = self.networks.portfolios(y)
        return w.reshape(n, -1), WITHDRAWAL_RATE


def _stack(groups):
    """One Scenarios object with the scenarios of all groups after each other."""
    first = groups[0]
    return Scenarios(torch.cat([g.returns for g in groups], dim=1), torch.cat([g.inflation for g in groups], dim=1),
                     torch.cat([g.market for g in groups], dim=1), first.asset_names, first.market_names)


def _per_group(x, n_models):
    """(N,) -> (M, N // M)."""
    return x.reshape(n_models, -1)


def _training_score(out, softness, n_models):
    """yield_mlp.py's score per network: mean log(1 + final wealth) - penalty * smooth ruin."""
    margins = out["margins"]
    ruined = (margins <= 0).to(DTYPE)
    earlier = torch.cat([torch.zeros_like(margins[:1]), ruined[:-1]]).cumsum(dim=0) > 0
    lowest = margins.masked_fill(earlier, float("inf")).min(dim=0).values
    closeness = _per_group(torch.sigmoid(-lowest / softness), n_models)
    final = _per_group(out["wealth"][-1], n_models)
    return torch.log1p(final).mean(dim=1) - RUIN_PENALTY * closeness.mean(dim=1)  # (M,)


def _check_score(out, n_models):
    """yield_mlp.py's check score per network: mean log(1 + final wealth) - penalty * ruin share."""
    ruined = _per_group((out["margins"] <= 0).any(dim=0).to(DTYPE), n_models)
    final = _per_group(out["wealth"][-1], n_models)
    return torch.log1p(final).mean(dim=1) - RUIN_PENALTY * ruined.mean(dim=1)


def train_yield_networks(train, n_sets=N_SETS, seed=0, log=print):
    """Train n_sets networks, each on its own bootstrap of `train`. Returns the trained Networks."""
    cfg = Config()
    # 1. one bootstrap set per network, from the train scenarios only
    fit_sets, check_sets = [], []
    for k in range(n_sets):
        fit_part, check_part = split_scenarios(train, VALIDATION_FRACTION, seed=seed + k)
        fit_sets.append(BlockBootstrap(fit_part, BLOCK_YEARS, seed=2 * (seed + k)).sample(N_TRAIN))
        check_sets.append(BlockBootstrap(check_part, BLOCK_YEARS, seed=2 * (seed + k) + 1).sample(N_VALIDATION))
    checks = _stack(check_sets)

    # 2. every network scales its inputs by the yields of its own train scenarios
    y = torch.stack([yields_of(s.market, s.market_names).reshape(-1, len(YIELDS)) for s in fit_sets])  # (M, ., 2)
    networks = Networks(n_sets, train.n_assets, y.mean(dim=1), y.std(dim=1), seed=seed)
    player = _Training(networks, n_sets)
    optimizer = torch.optim.Adam(networks.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)

    best_score = torch.full((n_sets,), -float("inf"), dtype=DTYPE)
    best = networks.snapshot()
    for step in range(1, TRAINING_STEPS + 1):
        softness = SOFTNESS_START * (SOFTNESS_END / SOFTNESS_START) ** ((step - 1) / (TRAINING_STEPS - 1))
        batch = _stack([s.subset(torch.randperm(N_TRAIN, generator=generator)[:SCENARIOS_PER_STEP])
                        for s in fit_sets])
        score = _training_score(simulate(player, batch, cfg), softness, n_sets)  # (M,)
        optimizer.zero_grad()
        (-score.sum()).backward()  # the networks share no numbers, so each gets its own gradient
        _clip_per_network(networks, max_norm=1.0)
        optimizer.step()

        if step % CHECK_EVERY == 0:  # keep, per network, the version that does best on its own check set
            with torch.no_grad():
                check = _check_score(simulate(player, checks, cfg), n_sets)
            better = check > best_score
            best_score = torch.where(better, check, best_score)
            for (bw, bb), (w, b) in zip(best, networks.layers):
                bw[better], bb[better] = w.detach()[better], b.detach()[better]
            if step % 100 == 0:
                log(f"  weights, step {step}: networks improved this check: {int(better.sum())} of {n_sets}")

    with torch.no_grad():
        for (bw, bb), (w, b) in zip(best, networks.layers):
            w.copy_(bw)
            b.copy_(bb)
    return networks


def _clip_per_network(networks, max_norm):
    """torch.nn.utils.clip_grad_norm_ for every network on its own, as if trained one by one."""
    params = networks.parameters()
    squared = sum(p.grad.pow(2).reshape(p.shape[0], -1).sum(dim=1) for p in params)  # (M,)
    scale = (max_norm / (squared.sqrt() + 1e-6)).clamp(max=1.0)
    for p in params:
        p.grad.mul_(scale.reshape(-1, *[1] * (p.dim() - 1)))


class AveragedYieldWeights:
    """Invests in the average of the portfolios of all trained networks (as AverageOfModels)."""

    def __init__(self, networks):
        self.networks = networks

    def parameters(self):
        return []  # already trained

    def weights(self, obs):
        with torch.no_grad():
            if obs.year == 0:
                return self.networks.first_year_portfolios().mean(dim=0)
            y = yields_of(obs.market_history[-1], obs.market_names)  # (N, 2)
            m = len(self.networks.layers[0][0])
            return self.networks.portfolios(y[None].expand(m, -1, -1)).mean(dim=0)  # (N, A)
