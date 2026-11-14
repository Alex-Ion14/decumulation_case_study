"""The investing models. Every model decides the portfolio weights; the withdrawal is fixed.

All models have the method ``act(obs)`` that utils.simulate expects, and return the weights of
the 7 assets and the fixed withdrawal rate (4% of the initial capital, as in the Bengen rule).

  FixedPolicy      a fixed mix, or a mix that changes with a fixed schedule (no learning)
  StaticPolicy     one learned mix, the same in every year and scenario
  MLPPolicy        a small neural network: this year's features -> weights
  RNNPolicy        a recurrent network (GRU or LSTM): remembers the earlier years of a scenario
  BayesianMLPPolicy  an MLPPolicy whose weights are uncertain: it learns a range for every
                     network weight, so we can draw many plausible networks and see where
                     they agree
  Ensemble         the average of several models, and how much they agree (the "confidence")
"""

import copy
import math

import torch
from torch import nn

from machine_learning_investing.features import Features
from utils import ASSET_NAMES, DTYPE

WITHDRAWAL_RATE = 0.04
N_ASSETS = len(ASSET_NAMES)
HIDDEN = 32


class Policy(nn.Module):
    rate = WITHDRAWAL_RATE

    def weights(self, obs):
        raise NotImplementedError

    def act(self, obs):
        return self.weights(obs), self.rate

    def fit_features(self, train):
        """Called once before training, with the train scenarios."""

    def extra_loss(self):
        """Something to add to the training loss (used by the Bayesian model)."""
        return 0.0


class FixedPolicy(Policy):
    """Fixed weights, or a straight line from ``start`` weights (year 0) to ``end`` weights (last year)."""

    def __init__(self, start, end=None):
        super().__init__()
        as_weights = lambda mix: torch.tensor([mix.get(name, 0.0) for name in ASSET_NAMES], dtype=DTYPE)
        self.start, self.end = as_weights(start), as_weights(end or start)

    def weights(self, obs):
        share = obs.year / (obs.n_years - 1)
        return (1 - share) * self.start + share * self.end


class StaticPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(N_ASSETS, dtype=DTYPE))  # starts at equal weights

    def weights(self, obs):
        return torch.softmax(self.logits, dim=0)


def _small_last_layer(layer):
    """Start close to equal weights in every situation, so training begins from a sensible mix."""
    with torch.no_grad():
        layer.weight.mul_(0.1)
        layer.bias.zero_()
    return layer


class MLPPolicy(Policy):
    def __init__(self, use_market=True):
        super().__init__()
        self.features = Features(self.rate, use_market)
        self.net = nn.Sequential(
            nn.Linear(self.features.size, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            _small_last_layer(nn.Linear(HIDDEN, N_ASSETS)),
        ).to(DTYPE)

    def fit_features(self, train):
        self.features.fit(train)

    def weights(self, obs):
        return torch.softmax(self.net(self.features(obs)), dim=1)


class RNNPolicy(Policy):
    """A recurrent network. Its memory is reset in year 0 and updated once per year."""

    def __init__(self, cell="gru", use_market=True):
        super().__init__()
        self.features = Features(self.rate, use_market)
        cell_class = {"gru": nn.GRUCell, "lstm": nn.LSTMCell}[cell]
        self.cell = cell_class(self.features.size, HIDDEN).to(DTYPE)
        self.head = _small_last_layer(nn.Linear(HIDDEN, N_ASSETS)).to(DTYPE)
        self.memory = None

    def fit_features(self, train):
        self.features.fit(train)

    def weights(self, obs):
        if obs.year == 0:
            self.memory = None  # a new scenario: forget everything
        self.memory = self.cell(self.features(obs), self.memory)
        hidden = self.memory[0] if isinstance(self.memory, tuple) else self.memory
        return torch.softmax(self.head(hidden), dim=1)


# ── Bayesian network ─────────────────────────────────────────────────────────

class BayesianLinear(nn.Module):
    """A linear layer in which every weight has an average and an uncertainty (a normal distribution)."""

    def __init__(self, n_in, n_out, prior_std, last=False):
        super().__init__()
        bound = (0.1 if last else 1.0) / math.sqrt(n_in)
        self.weight_mean = nn.Parameter(torch.empty(n_out, n_in, dtype=DTYPE).uniform_(-bound, bound))
        self.bias_mean = nn.Parameter(torch.zeros(n_out, dtype=DTYPE))
        self.weight_rho = nn.Parameter(torch.full((n_out, n_in), -4.0, dtype=DTYPE))  # std = softplus(rho)
        self.bias_rho = nn.Parameter(torch.full((n_out,), -4.0, dtype=DTYPE))
        self.prior_std = prior_std
        self.noise = None  # None: use the average weights

    def draw(self):
        self.noise = (torch.randn_like(self.weight_mean), torch.randn_like(self.bias_mean))

    def forward(self, x):
        weight, bias = self.weight_mean, self.bias_mean
        if self.noise is not None:
            weight = weight + nn.functional.softplus(self.weight_rho) * self.noise[0]
            bias = bias + nn.functional.softplus(self.bias_rho) * self.noise[1]
        return nn.functional.linear(x, weight, bias)

    def kl(self):
        """Distance between the learned weight distributions and the prior N(0, prior_std)."""
        total = 0.0
        for mean, rho in ((self.weight_mean, self.weight_rho), (self.bias_mean, self.bias_rho)):
            std = nn.functional.softplus(rho)
            total = total + (torch.log(self.prior_std / std)
                             + (std ** 2 + mean ** 2) / (2 * self.prior_std ** 2) - 0.5).sum()
        return total


class BayesianMLPPolicy(Policy):
    """An MLP trained with variational inference ("Bayes by backprop").

    During training a new network is drawn before every simulation, and the loss gets a
    penalty (the KL term) for weight distributions that are far from the prior. Weights that
    matter for the result become precise; weights the data says little about stay uncertain.

    kl_weight: how strongly the prior counts against the data; larger values give a wider,
    more cautious posterior. The textbook value is 1 / (number of independent scenarios
    behind the train data) = 1/122, but our score is not a probability of the data, and with
    that value the prior simply wins: every weight keeps the prior's uncertainty and the
    draws are random portfolios. 0.0003 was chosen on validation scenarios: the draws then
    still score well and the uncertainty differs between weights. Because this number is a
    choice, the *level* of the Bayesian confidence is too; where it is higher or lower is
    what carries information.
    """

    def __init__(self, use_market=True, prior_std=0.5, kl_weight=3e-4):
        super().__init__()
        self.features = Features(self.rate, use_market)
        self.layers = nn.ModuleList([
            BayesianLinear(self.features.size, HIDDEN, prior_std),
            BayesianLinear(HIDDEN, HIDDEN, prior_std),
            BayesianLinear(HIDDEN, N_ASSETS, prior_std, last=True),
        ])
        self.kl_weight = kl_weight

    def fit_features(self, train):
        self.features.fit(train)

    def draw(self):
        for layer in self.layers:
            layer.draw()

    def use_average(self):
        for layer in self.layers:
            layer.noise = None

    def extra_loss(self):
        return self.kl_weight * sum(layer.kl() for layer in self.layers)

    def weights(self, obs):
        x = self.features(obs)
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return torch.softmax(self.layers[-1](x), dim=1)

    def draw_members(self, n, seed=0):
        """n networks drawn from the learned weight distributions, each with fixed weights."""
        torch.manual_seed(seed)
        members = []
        for _ in range(n):
            member = copy.deepcopy(self)
            member.draw()
            members.append(member)
        return members


# ── Ensemble ─────────────────────────────────────────────────────────────────

class Ensemble(Policy):
    """Invests in the average of the members' portfolios and records how much they agree.

    agreement = the share of the portfolio that is the same in a member's portfolio and in
    the average portfolio, averaged over the members. 100% means that all members propose
    exactly the same portfolio; lower values mean that the proposal depends on which model
    (or which resample of the data) you ask.
    """

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.agreement = []  # one (N,) tensor per year of the last simulation
        self.average_weights = []  # one (N, A) tensor per year

    def weights(self, obs):
        if obs.year == 0:
            self.agreement, self.average_weights = [], []
        proposals = torch.stack([member.act(obs)[0].expand(len(obs.wealth), N_ASSETS) for member in self.members])
        average = proposals.mean(dim=0)
        different = 0.5 * (proposals - average).abs().sum(dim=2)  # (members, N) share that differs
        self.agreement.append((1 - different.mean(dim=0)).detach())
        self.average_weights.append(average.detach())
        return average
