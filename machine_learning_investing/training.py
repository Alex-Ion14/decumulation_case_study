"""Training and scoring of the investing models.

The goal: with a fixed withdrawal of 4% of the initial capital per year (in real terms), end
with as much wealth as possible without running out of money. As one number:

    score = average of log(1 + final wealth)  -  ruin probability

The logarithm makes the score care about the typical scenario and not only about a few
scenarios with a huge final wealth (doubling your wealth always adds the same amount), and a
ruined scenario counts twice: its final wealth is 0 and it adds to the ruin probability.

A model is trained by gradient ascent on this score through utils.simulate. Ruin is a yes/no
event without a useful gradient, so during training it is replaced by a smooth version, as in
strategies/bengen_optimised.py. Training stops at the step with the best score on the
validation scenarios.
"""

import copy

import torch

from utils import simulate

RUIN_PENALTY = 1.0
STEPS = 300
EVALUATE_EVERY = 20
TAU_START, TAU_END = 0.1, 0.01  # smoothness of the ruin approximation at the start and the end
WEIGHT_DECAY = 1e-4
GROWTH_ASSETS = [0, 1, 2]  # US equity, EM equity, US REIT


class _Recorder:
    """Passes ``act`` on to a model and remembers the weights it chose."""

    def __init__(self, policy):
        self.policy, self.weights = policy, []

    def act(self, obs):
        weights, rate = self.policy.act(obs)
        self.weights.append(torch.as_tensor(weights).expand(len(obs.wealth), -1))
        return weights, rate


def outcomes(policy, scenarios, cfg):
    """Simulate without training. Returns per-scenario results and the weights of every year."""
    policy.eval()
    recorder = _Recorder(policy)
    with torch.no_grad():
        out = simulate(recorder, scenarios, cfg)
    weights = torch.stack(recorder.weights)  # (T, N, A)
    drifted = weights[:-1] * (1 + scenarios.returns[:-1])
    drifted = drifted / drifted.sum(dim=2, keepdim=True)
    return {
        "final_wealth": out["wealth"][-1],
        "ruined": (out["margins"] <= 0).any(dim=0),
        "wealth": out["wealth"],
        "weights": weights,
        "turnover": (weights[1:] - drifted).abs().sum(dim=2).mean(dim=0) / 2,  # (N,) share of wealth sold per year
    }


def metrics(policy, scenarios, cfg):
    out = outcomes(policy, scenarios, cfg)
    final = out["final_wealth"]
    ruin_prob = out["ruined"].double().mean().item()
    return {
        "score": torch.log1p(final).mean().item() - RUIN_PENALTY * ruin_prob,
        "ruin_prob": ruin_prob,
        "median_final_wealth": final.median().item(),
        "mean_final_wealth": final.mean().item(),
        "p10_final_wealth": final.quantile(0.10).item(),
        "growth_share": out["weights"][:, :, GROWTH_ASSETS].sum(dim=2).mean().item(),
        "turnover": out["turnover"].mean().item(),
    }


def smooth_score(out, tau):
    """The score with a smooth version of ruin, for training."""
    margins = out["margins"]  # (T, N) wealth left after the withdrawal; <= 0 means ruined
    # After a ruin the wealth stays 0 whatever the model does. Only look at the years up to
    # and including the first ruin, so the gradient points at the year where it went wrong.
    ruined_before = torch.cat([torch.zeros_like(margins[:1]), (margins[:-1] <= 0).to(margins.dtype)])
    already_ruined = ruined_before.cumsum(dim=0) > 0
    lowest_margin = margins.masked_fill(already_ruined, float("inf")).min(dim=0).values
    smooth_ruin_prob = torch.sigmoid(-lowest_margin / tau).mean()
    return torch.log1p(out["wealth"][-1]).mean() - RUIN_PENALTY * smooth_ruin_prob


def train(policy, train_scenarios, validation_scenarios, cfg, learning_rate=0.01, steps=STEPS, seed=0,
          batch_size=None, min_steps=0, verbose=False):
    """Train a model on the train scenarios; keep the version that scores best on validation.

    batch_size: train every step on a random part of the train scenarios (None: on all).
    min_steps: never keep a version from before this step (the Bayesian model needs time
               for its uncertainties to settle).
    Returns the model (also changed in place) and the validation score it was chosen on.
    """
    torch.manual_seed(seed)
    policy.fit_features(train_scenarios)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if not parameters:  # nothing to learn (a fixed rule)
        return policy, metrics(policy, validation_scenarios, cfg)["score"]
    bayesian = hasattr(policy, "draw")
    if bayesian:
        # The uncertainties learn 10 times faster than the average weights, so they have
        # settled before the average network starts to overfit. The prior does the
        # regularising here, so no weight decay.
        uncertainties = [p for name, p in policy.named_parameters() if name.endswith("_rho")]
        averages = [p for name, p in policy.named_parameters() if not name.endswith("_rho")]
        optimizer = torch.optim.Adam([{"params": averages}, {"params": uncertainties, "lr": 10 * learning_rate}],
                                     lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    best_score, best_state = -float("inf"), None

    for step in range(steps):
        tau = TAU_START * (TAU_END / TAU_START) ** (step / (steps - 1))
        batch = train_scenarios
        if batch_size is not None:
            batch = train_scenarios.subset(torch.randperm(train_scenarios.n_scenarios)[:batch_size])
        policy.train()
        if bayesian:
            policy.draw()  # a new plausible network for this simulation
        loss = -smooth_score(simulate(policy, batch, cfg), tau) + policy.extra_loss()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if (step + 1) % EVALUATE_EVERY == 0 and step + 1 >= min_steps:
            if bayesian:
                policy.use_average()
            score = metrics(policy, validation_scenarios, cfg)["score"]
            if score > best_score:
                best_score, best_state = score, copy.deepcopy(policy.state_dict())
            if verbose:
                print(f"    step {step + 1}: validation score {score:.4f}")

    policy.load_state_dict(best_state)
    if bayesian:
        policy.use_average()
    return policy, best_score
