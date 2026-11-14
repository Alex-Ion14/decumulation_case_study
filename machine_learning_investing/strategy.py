"""Training-selected GRU policy for the case evaluator.

python main.py machine_learning_investing/strategy.py --no-frontier
"""

import torch

from machine_learning_investing.policies import SEED, NeuralAllocation


class Strategy:
    def fit(self, train, cfg):
        # CPU batches in this case are small, and one thread avoids pool overhead.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.policy = NeuralAllocation("gru", train, cfg, SEED)
        finally:
            torch.set_num_threads(threads)

    def act(self, obs):
        return self.policy.act(obs)
