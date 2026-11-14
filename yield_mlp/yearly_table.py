'''The average portfolio of the yield model for each of the 30 years.

THIS FILE WAS WRITTEN BY yield_mlp/make_yearly_table.py. Do not change the numbers by hand.

IMPORTANT: this is NOT the yield model. It is the average of what the yield model holds in
each year, over the 153 pool scenarios. It depends on nothing but the year, so it cannot
react to the bond yields. On the pool scenarios it ends with a median final wealth of about
2.2 times the starting capital, against about 3.5 for the yield table in
hardcoded_strategy.py.

Use it as a strategy of the case:

    python main.py yield_mlp/yearly_table.py --no-frontier
'''

import torch

WITHDRAWAL_RATE = 0.04

# The order of the 7 numbers in every portfolio below.
ASSETS = ['US equity', 'EM equity', 'US REIT', 'US Treasury 5Y', 'US Treasury 30Y', 'US TIPS 10Y', 'Cash']

# YEARLY_PORTFOLIOS[0] is the portfolio for the first year, YEARLY_PORTFOLIOS[29] for the last.
# Because of rounding the weights can add up to 0.999 or 1.001; the simulator scales them to 1.
YEARLY_PORTFOLIOS = [
    [0.214, 0.373, 0.176, 0.003, 0.230, 0.003, 0.002],  # year 1
    [0.216, 0.319, 0.173, 0.026, 0.256, 0.006, 0.004],  # year 2
    [0.212, 0.320, 0.171, 0.027, 0.261, 0.006, 0.004],  # year 3
    [0.213, 0.319, 0.171, 0.028, 0.260, 0.006, 0.004],  # year 4
    [0.215, 0.318, 0.173, 0.029, 0.254, 0.006, 0.004],  # year 5
    [0.216, 0.319, 0.173, 0.028, 0.255, 0.006, 0.004],  # year 6
    [0.217, 0.320, 0.173, 0.028, 0.252, 0.006, 0.004],  # year 7
    [0.220, 0.323, 0.176, 0.026, 0.246, 0.005, 0.003],  # year 8
    [0.219, 0.320, 0.175, 0.027, 0.250, 0.005, 0.004],  # year 9
    [0.225, 0.326, 0.181, 0.024, 0.236, 0.005, 0.004],  # year 10
    [0.227, 0.325, 0.181, 0.024, 0.235, 0.005, 0.004],  # year 11
    [0.226, 0.320, 0.179, 0.023, 0.243, 0.005, 0.004],  # year 12
    [0.218, 0.317, 0.173, 0.028, 0.253, 0.006, 0.004],  # year 13
    [0.213, 0.312, 0.169, 0.031, 0.265, 0.006, 0.004],  # year 14
    [0.208, 0.307, 0.165, 0.032, 0.277, 0.007, 0.004],  # year 15
    [0.208, 0.310, 0.166, 0.028, 0.277, 0.006, 0.004],  # year 16
    [0.211, 0.312, 0.167, 0.028, 0.273, 0.006, 0.004],  # year 17
    [0.215, 0.315, 0.171, 0.027, 0.262, 0.006, 0.004],  # year 18
    [0.214, 0.316, 0.170, 0.026, 0.264, 0.006, 0.004],  # year 19
    [0.220, 0.317, 0.175, 0.027, 0.250, 0.006, 0.004],  # year 20
    [0.218, 0.316, 0.173, 0.026, 0.257, 0.006, 0.004],  # year 21
    [0.216, 0.316, 0.171, 0.026, 0.260, 0.006, 0.004],  # year 22
    [0.221, 0.317, 0.173, 0.026, 0.253, 0.006, 0.004],  # year 23
    [0.221, 0.317, 0.175, 0.026, 0.251, 0.006, 0.004],  # year 24
    [0.222, 0.320, 0.176, 0.026, 0.248, 0.006, 0.004],  # year 25
    [0.220, 0.319, 0.174, 0.026, 0.252, 0.006, 0.004],  # year 26
    [0.228, 0.322, 0.179, 0.024, 0.239, 0.005, 0.004],  # year 27
    [0.230, 0.320, 0.180, 0.024, 0.237, 0.005, 0.004],  # year 28
    [0.227, 0.319, 0.177, 0.024, 0.243, 0.005, 0.004],  # year 29
    [0.223, 0.319, 0.175, 0.026, 0.249, 0.005, 0.004],  # year 30
]


class Strategy:
    '''The strategy in the form that main.py of the case expects.'''

    def fit(self, train, cfg):
        pass  # nothing to learn: the table is fixed

    def act(self, obs):
        # obs.year counts from 0 (first year) to 29 (last year).
        return torch.tensor(YEARLY_PORTFOLIOS[obs.year], dtype=torch.float64), WITHDRAWAL_RATE
