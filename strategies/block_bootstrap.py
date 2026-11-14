"""Block bootstrap, copied from the data team's bootstrap/sampler.py so that strategies/run.py
does not depend on the bootstrap/ folder. Keep the two in step if the data team changes theirs.

Every new scenario is made of *blocks*: a few consecutive years copied from one existing
scenario, followed by a few consecutive years copied from another, and so on until the
scenario is full. A copied year always brings everything along (all asset returns, inflation
and the extra market variables), so the way the assets move together is kept. Copying several
years in a row keeps most of the year-to-year patterns as well (for example, interest rates
that stay low for a long time).

    from strategies.block_bootstrap import BlockBootstrap, split_scenarios

    fit_part, check_part = split_scenarios(train, holdout_fraction=0.2, seed=0)
    sampler = BlockBootstrap(fit_part, block_years=5, seed=0)
    for step in range(STEPS):
        batch = sampler.sample(256)           # 256 new scenarios, different every step
        out = simulate(strategy, batch, cfg)  # ... train on them as usual
    evaluate(strategy, check_part, cfg)       # check on scenarios that were never cut up

Things to keep in mind:
  - The bootstrap invents nothing: every year is a copy. It helps against memorising the
    train scenarios, but it cannot show a strategy markets that are not in the data.
  - Where two blocks meet (a *seam*), the market variables jump: the yield at the end of one
    block has nothing to do with the bond return at the start of the next. ``sample`` can
    return the seam positions for strategies that use ``market_history``.
  - Split first, bootstrap second (``split_scenarios``). Otherwise pieces of the same scenario
    end up both in the training data and in the data you check on.
"""

import torch

from utils import Scenarios


def split_scenarios(scenarios, holdout_fraction=0.2, seed=0):
    """Randomly split whole scenarios into a part to bootstrap from and a part to check on."""
    n = scenarios.n_scenarios
    n_holdout = round(n * holdout_fraction)
    if not 0 < n_holdout < n:
        raise ValueError("holdout_fraction must leave at least one scenario on each side")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return scenarios.subset(order[n_holdout:]), scenarios.subset(order[:n_holdout])


class BlockBootstrap:
    """Draws new scenarios from blocks of consecutive years of the given scenarios.

    block_years : the length of a block. Between 5 and 10 works well for this data: single
                  years (block_years=1) lose the patterns between years, and very long blocks
                  give scenarios that are nearly copies of the originals.
    seed        : for reproducible draws; None gives different scenarios on every run.
    """

    def __init__(self, scenarios, block_years=5, seed=None):
        if not 1 <= block_years <= scenarios.n_years:
            raise ValueError(f"block_years must be between 1 and {scenarios.n_years}")
        self.scenarios = scenarios
        self.block_years = block_years
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def sample(self, n, return_seams=False):
        """Return ``n`` new scenarios, in the same format as ``load_scenarios``.

        With ``return_seams=True`` it also returns a (T, n) tensor that is True for every year
        that starts a new block (the first year excluded).
        """
        source, n_years, length = self.scenarios, self.scenarios.n_years, self.block_years

        def random_integers(high, shape):
            return torch.randint(high, shape, generator=self.generator)

        # Every new scenario starts somewhere inside its first block, so the seams do not fall
        # in the same years in every scenario.
        offset = random_integers(length, (n,))
        position = torch.arange(n_years)[:, None] + offset  # (T, n) years since the first block began
        block = position // length  # which block a year belongs to
        year_in_block = position % length

        # Every block copies `length` consecutive years from one randomly chosen scenario.
        n_blocks = int(block.max()) + 1
        block_scenario = random_integers(source.n_scenarios, (n_blocks, n))
        block_first_year = random_integers(n_years - length + 1, (n_blocks, n))

        from_scenario = block_scenario.gather(0, block)  # (T, n) the scenario every year is copied from
        from_year = block_first_year.gather(0, block) + year_in_block  # (T, n) ... and the year

        new = Scenarios(
            returns=source.returns[from_year, from_scenario],
            inflation=source.inflation[from_year, from_scenario],
            market=source.market[from_year, from_scenario],
            asset_names=source.asset_names,
            market_names=source.market_names,
        )
        if return_seams:
            seams = year_in_block == 0
            seams[0] = False
            return new, seams
        return new
