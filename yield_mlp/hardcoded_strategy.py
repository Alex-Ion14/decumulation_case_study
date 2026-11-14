'''The yield model as a table: no neural network and no training needed.

THIS FILE WAS WRITTEN BY yield_mlp/yield_mlp.py. Do not change the numbers by hand; run
that file again to make a new table.

What it does: every year it looks at the 5-year and the 30-year US Treasury yield at the end
of last year and looks up the portfolio that belongs to them. The portfolios in the table
are the average of the portfolios proposed by 20 small neural networks.
The withdrawal is fixed at 4% of the starting capital.

Use it as a strategy of the case:

    python main.py yield_mlp/hardcoded_strategy.py --no-frontier

or copy the tables and the function look_up_portfolio into your own strategies/run.py.
'''

import torch

WITHDRAWAL_RATE = 0.04

# The order of the 7 numbers in every portfolio below.
ASSETS = ['US equity', 'EM equity', 'US REIT', 'US Treasury 5Y', 'US Treasury 30Y', 'US TIPS 10Y', 'Cash']

# The portfolio for the first year, when no yield of "last year" is known yet.
FIRST_YEAR_PORTFOLIO = [0.214, 0.373, 0.176, 0.003, 0.230, 0.003, 0.002]

# The yields (in percent) for which the table has an entry. The same steps are used for the
# 5-year yield and for the 30-year yield.
TABLE_YIELDS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

# PORTFOLIOS[i][j] is the portfolio when the 5-year yield is TABLE_YIELDS[i] and the 30-year
# yield is TABLE_YIELDS[j]. Every portfolio is 7 weights in the order of ASSETS. Because of
# rounding they can add up to 0.999 or 1.001; the simulator scales them to exactly 1.
PORTFOLIOS = [
    [  # 5-year yield 0%
        [0.420, 0.298, 0.259, 0.003, 0.012, 0.003, 0.006],  # 30-year yield 0%
        [0.385, 0.337, 0.257, 0.001, 0.014, 0.002, 0.003],  # 30-year yield 1%
        [0.350, 0.366, 0.261, 0.001, 0.019, 0.001, 0.002],  # 30-year yield 2%
        [0.336, 0.376, 0.260, 0.001, 0.025, 0.001, 0.002],  # 30-year yield 3%
        [0.330, 0.380, 0.254, 0.001, 0.033, 0.001, 0.002],  # 30-year yield 4%
        [0.326, 0.383, 0.243, 0.001, 0.045, 0.001, 0.002],  # 30-year yield 5%
        [0.318, 0.391, 0.223, 0.001, 0.064, 0.001, 0.002],  # 30-year yield 6%
        [0.300, 0.397, 0.201, 0.001, 0.096, 0.002, 0.002],  # 30-year yield 7%
        [0.271, 0.390, 0.178, 0.002, 0.154, 0.002, 0.003],  # 30-year yield 8%
        [0.237, 0.368, 0.154, 0.003, 0.232, 0.003, 0.004],  # 30-year yield 9%
        [0.206, 0.341, 0.132, 0.004, 0.309, 0.004, 0.004],  # 30-year yield 10%
        [0.179, 0.316, 0.114, 0.004, 0.377, 0.005, 0.005],  # 30-year yield 11%
        [0.158, 0.293, 0.099, 0.005, 0.434, 0.005, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 1%
        [0.412, 0.285, 0.272, 0.004, 0.013, 0.005, 0.008],  # 30-year yield 0%
        [0.387, 0.328, 0.261, 0.002, 0.016, 0.002, 0.004],  # 30-year yield 1%
        [0.344, 0.367, 0.262, 0.001, 0.022, 0.001, 0.002],  # 30-year yield 2%
        [0.325, 0.381, 0.260, 0.001, 0.030, 0.001, 0.002],  # 30-year yield 3%
        [0.319, 0.384, 0.250, 0.001, 0.044, 0.001, 0.002],  # 30-year yield 4%
        [0.313, 0.388, 0.232, 0.001, 0.063, 0.001, 0.002],  # 30-year yield 5%
        [0.297, 0.398, 0.205, 0.001, 0.096, 0.001, 0.002],  # 30-year yield 6%
        [0.266, 0.392, 0.179, 0.002, 0.157, 0.002, 0.002],  # 30-year yield 7%
        [0.227, 0.369, 0.150, 0.002, 0.246, 0.002, 0.002],  # 30-year yield 8%
        [0.191, 0.337, 0.124, 0.003, 0.339, 0.003, 0.003],  # 30-year yield 9%
        [0.162, 0.306, 0.103, 0.004, 0.417, 0.004, 0.004],  # 30-year yield 10%
        [0.140, 0.279, 0.088, 0.005, 0.479, 0.005, 0.004],  # 30-year yield 11%
        [0.124, 0.257, 0.078, 0.006, 0.525, 0.005, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 2%
        [0.388, 0.269, 0.294, 0.009, 0.016, 0.011, 0.014],  # 30-year yield 0%
        [0.374, 0.318, 0.274, 0.003, 0.020, 0.004, 0.005],  # 30-year yield 1%
        [0.332, 0.366, 0.266, 0.002, 0.030, 0.002, 0.003],  # 30-year yield 2%
        [0.303, 0.389, 0.257, 0.001, 0.047, 0.001, 0.002],  # 30-year yield 3%
        [0.294, 0.392, 0.238, 0.001, 0.072, 0.001, 0.002],  # 30-year yield 4%
        [0.281, 0.397, 0.208, 0.001, 0.110, 0.001, 0.002],  # 30-year yield 5%
        [0.254, 0.393, 0.177, 0.002, 0.170, 0.002, 0.002],  # 30-year yield 6%
        [0.216, 0.370, 0.146, 0.002, 0.262, 0.002, 0.002],  # 30-year yield 7%
        [0.175, 0.331, 0.115, 0.003, 0.370, 0.003, 0.002],  # 30-year yield 8%
        [0.144, 0.292, 0.093, 0.004, 0.460, 0.004, 0.003],  # 30-year yield 9%
        [0.123, 0.259, 0.078, 0.005, 0.527, 0.004, 0.003],  # 30-year yield 10%
        [0.108, 0.235, 0.069, 0.006, 0.574, 0.005, 0.004],  # 30-year yield 11%
        [0.098, 0.216, 0.063, 0.007, 0.607, 0.006, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 3%
        [0.336, 0.245, 0.296, 0.049, 0.024, 0.026, 0.025],  # 30-year yield 0%
        [0.337, 0.297, 0.297, 0.014, 0.034, 0.011, 0.010],  # 30-year yield 1%
        [0.302, 0.357, 0.274, 0.004, 0.056, 0.004, 0.004],  # 30-year yield 2%
        [0.263, 0.393, 0.241, 0.002, 0.096, 0.002, 0.002],  # 30-year yield 3%
        [0.245, 0.393, 0.204, 0.002, 0.152, 0.002, 0.002],  # 30-year yield 4%
        [0.226, 0.382, 0.168, 0.002, 0.218, 0.002, 0.002],  # 30-year yield 5%
        [0.199, 0.360, 0.140, 0.003, 0.294, 0.002, 0.002],  # 30-year yield 6%
        [0.164, 0.326, 0.110, 0.003, 0.391, 0.003, 0.002],  # 30-year yield 7%
        [0.131, 0.283, 0.086, 0.004, 0.489, 0.004, 0.002],  # 30-year yield 8%
        [0.108, 0.245, 0.071, 0.005, 0.563, 0.004, 0.003],  # 30-year yield 9%
        [0.094, 0.217, 0.062, 0.006, 0.613, 0.005, 0.003],  # 30-year yield 10%
        [0.084, 0.197, 0.056, 0.007, 0.646, 0.006, 0.004],  # 30-year yield 11%
        [0.077, 0.183, 0.052, 0.008, 0.670, 0.007, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 4%
        [0.257, 0.213, 0.227, 0.180, 0.045, 0.044, 0.036],  # 30-year yield 0%
        [0.252, 0.247, 0.241, 0.145, 0.070, 0.027, 0.019],  # 30-year yield 1%
        [0.243, 0.307, 0.257, 0.057, 0.118, 0.011, 0.007],  # 30-year yield 2%
        [0.215, 0.352, 0.215, 0.010, 0.200, 0.004, 0.003],  # 30-year yield 3%
        [0.193, 0.356, 0.161, 0.004, 0.280, 0.003, 0.002],  # 30-year yield 4%
        [0.174, 0.347, 0.133, 0.004, 0.336, 0.003, 0.002],  # 30-year yield 5%
        [0.151, 0.329, 0.110, 0.004, 0.400, 0.004, 0.002],  # 30-year yield 6%
        [0.122, 0.294, 0.086, 0.005, 0.487, 0.004, 0.002],  # 30-year yield 7%
        [0.097, 0.248, 0.068, 0.006, 0.573, 0.005, 0.003],  # 30-year yield 8%
        [0.081, 0.211, 0.057, 0.007, 0.635, 0.005, 0.003],  # 30-year yield 9%
        [0.072, 0.186, 0.050, 0.007, 0.675, 0.006, 0.003],  # 30-year yield 10%
        [0.065, 0.169, 0.046, 0.008, 0.700, 0.007, 0.004],  # 30-year yield 11%
        [0.061, 0.158, 0.044, 0.009, 0.717, 0.007, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 5%
        [0.193, 0.184, 0.168, 0.266, 0.085, 0.060, 0.044],  # 30-year yield 0%
        [0.182, 0.207, 0.165, 0.256, 0.120, 0.043, 0.027],  # 30-year yield 1%
        [0.169, 0.237, 0.159, 0.218, 0.181, 0.023, 0.012],  # 30-year yield 2%
        [0.155, 0.273, 0.163, 0.110, 0.283, 0.011, 0.005],  # 30-year yield 3%
        [0.148, 0.311, 0.136, 0.014, 0.382, 0.006, 0.003],  # 30-year yield 4%
        [0.131, 0.314, 0.107, 0.007, 0.433, 0.005, 0.003],  # 30-year yield 5%
        [0.114, 0.300, 0.087, 0.007, 0.485, 0.005, 0.003],  # 30-year yield 6%
        [0.092, 0.264, 0.069, 0.007, 0.560, 0.005, 0.003],  # 30-year yield 7%
        [0.073, 0.220, 0.055, 0.008, 0.636, 0.006, 0.003],  # 30-year yield 8%
        [0.061, 0.185, 0.047, 0.008, 0.689, 0.006, 0.003],  # 30-year yield 9%
        [0.055, 0.162, 0.042, 0.009, 0.721, 0.007, 0.004],  # 30-year yield 10%
        [0.051, 0.148, 0.039, 0.010, 0.741, 0.008, 0.004],  # 30-year yield 11%
        [0.048, 0.139, 0.037, 0.010, 0.753, 0.008, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 6%
        [0.144, 0.155, 0.129, 0.311, 0.127, 0.080, 0.054],  # 30-year yield 0%
        [0.131, 0.168, 0.121, 0.304, 0.176, 0.063, 0.036],  # 30-year yield 1%
        [0.122, 0.188, 0.111, 0.280, 0.240, 0.040, 0.020],  # 30-year yield 2%
        [0.108, 0.207, 0.103, 0.218, 0.336, 0.019, 0.009],  # 30-year yield 3%
        [0.100, 0.240, 0.099, 0.115, 0.429, 0.012, 0.005],  # 30-year yield 4%
        [0.100, 0.274, 0.090, 0.020, 0.504, 0.008, 0.004],  # 30-year yield 5%
        [0.089, 0.268, 0.072, 0.010, 0.551, 0.007, 0.003],  # 30-year yield 6%
        [0.073, 0.238, 0.057, 0.009, 0.613, 0.007, 0.003],  # 30-year yield 7%
        [0.058, 0.197, 0.046, 0.009, 0.680, 0.007, 0.003],  # 30-year yield 8%
        [0.048, 0.165, 0.040, 0.010, 0.727, 0.007, 0.004],  # 30-year yield 9%
        [0.043, 0.144, 0.036, 0.010, 0.754, 0.008, 0.004],  # 30-year yield 10%
        [0.041, 0.132, 0.034, 0.011, 0.770, 0.008, 0.004],  # 30-year yield 11%
        [0.039, 0.124, 0.033, 0.011, 0.779, 0.009, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 7%
        [0.103, 0.124, 0.093, 0.351, 0.169, 0.099, 0.060],  # 30-year yield 0%
        [0.094, 0.130, 0.085, 0.350, 0.214, 0.082, 0.044],  # 30-year yield 1%
        [0.087, 0.141, 0.079, 0.335, 0.272, 0.058, 0.028],  # 30-year yield 2%
        [0.081, 0.160, 0.076, 0.282, 0.354, 0.032, 0.014],  # 30-year yield 3%
        [0.073, 0.189, 0.072, 0.196, 0.444, 0.018, 0.008],  # 30-year yield 4%
        [0.073, 0.216, 0.069, 0.109, 0.515, 0.013, 0.006],  # 30-year yield 5%
        [0.072, 0.233, 0.064, 0.023, 0.595, 0.010, 0.004],  # 30-year yield 6%
        [0.060, 0.213, 0.050, 0.012, 0.651, 0.009, 0.004],  # 30-year yield 7%
        [0.048, 0.178, 0.040, 0.011, 0.709, 0.008, 0.004],  # 30-year yield 8%
        [0.041, 0.149, 0.035, 0.011, 0.751, 0.008, 0.004],  # 30-year yield 9%
        [0.036, 0.131, 0.032, 0.012, 0.777, 0.009, 0.004],  # 30-year yield 10%
        [0.034, 0.120, 0.030, 0.012, 0.791, 0.009, 0.004],  # 30-year yield 11%
        [0.033, 0.113, 0.029, 0.012, 0.798, 0.009, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 8%
        [0.078, 0.100, 0.070, 0.383, 0.196, 0.110, 0.063],  # 30-year yield 0%
        [0.072, 0.104, 0.065, 0.382, 0.236, 0.093, 0.048],  # 30-year yield 1%
        [0.068, 0.111, 0.061, 0.369, 0.286, 0.071, 0.034],  # 30-year yield 2%
        [0.064, 0.125, 0.060, 0.331, 0.351, 0.048, 0.022],  # 30-year yield 3%
        [0.061, 0.150, 0.060, 0.256, 0.434, 0.027, 0.012],  # 30-year yield 4%
        [0.057, 0.173, 0.055, 0.185, 0.505, 0.017, 0.007],  # 30-year yield 5%
        [0.057, 0.187, 0.053, 0.096, 0.587, 0.014, 0.006],  # 30-year yield 6%
        [0.052, 0.187, 0.047, 0.024, 0.675, 0.011, 0.005],  # 30-year yield 7%
        [0.043, 0.162, 0.037, 0.015, 0.729, 0.010, 0.005],  # 30-year yield 8%
        [0.036, 0.138, 0.032, 0.013, 0.767, 0.010, 0.004],  # 30-year yield 9%
        [0.032, 0.121, 0.029, 0.013, 0.791, 0.010, 0.005],  # 30-year yield 10%
        [0.030, 0.111, 0.027, 0.013, 0.804, 0.010, 0.005],  # 30-year yield 11%
        [0.029, 0.104, 0.027, 0.013, 0.812, 0.010, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 9%
        [0.064, 0.085, 0.057, 0.403, 0.211, 0.116, 0.063],  # 30-year yield 0%
        [0.060, 0.089, 0.054, 0.399, 0.247, 0.100, 0.051],  # 30-year yield 1%
        [0.057, 0.094, 0.051, 0.386, 0.291, 0.082, 0.039],  # 30-year yield 2%
        [0.054, 0.104, 0.050, 0.356, 0.345, 0.062, 0.028],  # 30-year yield 3%
        [0.053, 0.122, 0.051, 0.303, 0.413, 0.040, 0.018],  # 30-year yield 4%
        [0.049, 0.141, 0.049, 0.234, 0.492, 0.024, 0.010],  # 30-year yield 5%
        [0.045, 0.151, 0.044, 0.177, 0.557, 0.018, 0.008],  # 30-year yield 6%
        [0.044, 0.152, 0.040, 0.084, 0.658, 0.014, 0.006],  # 30-year yield 7%
        [0.038, 0.145, 0.036, 0.025, 0.738, 0.012, 0.005],  # 30-year yield 8%
        [0.033, 0.128, 0.030, 0.017, 0.777, 0.011, 0.005],  # 30-year yield 9%
        [0.029, 0.113, 0.027, 0.015, 0.800, 0.011, 0.005],  # 30-year yield 10%
        [0.027, 0.103, 0.026, 0.015, 0.813, 0.010, 0.005],  # 30-year yield 11%
        [0.026, 0.098, 0.025, 0.015, 0.821, 0.011, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 10%
        [0.056, 0.076, 0.049, 0.415, 0.221, 0.119, 0.063],  # 30-year yield 0%
        [0.053, 0.079, 0.047, 0.410, 0.253, 0.106, 0.053],  # 30-year yield 1%
        [0.050, 0.084, 0.045, 0.395, 0.293, 0.090, 0.044],  # 30-year yield 2%
        [0.048, 0.091, 0.044, 0.371, 0.339, 0.072, 0.034],  # 30-year yield 3%
        [0.046, 0.103, 0.045, 0.334, 0.394, 0.053, 0.024],  # 30-year yield 4%
        [0.044, 0.118, 0.044, 0.277, 0.469, 0.034, 0.015],  # 30-year yield 5%
        [0.040, 0.126, 0.040, 0.221, 0.541, 0.023, 0.010],  # 30-year yield 6%
        [0.036, 0.125, 0.035, 0.163, 0.613, 0.019, 0.008],  # 30-year yield 7%
        [0.034, 0.121, 0.032, 0.076, 0.715, 0.015, 0.007],  # 30-year yield 8%
        [0.031, 0.117, 0.030, 0.025, 0.779, 0.013, 0.006],  # 30-year yield 9%
        [0.027, 0.106, 0.026, 0.019, 0.804, 0.012, 0.006],  # 30-year yield 10%
        [0.025, 0.098, 0.024, 0.017, 0.819, 0.011, 0.006],  # 30-year yield 11%
        [0.024, 0.092, 0.023, 0.016, 0.827, 0.011, 0.006],  # 30-year yield 12%
    ],
    [  # 5-year yield 11%
        [0.050, 0.069, 0.044, 0.423, 0.227, 0.122, 0.064],  # 30-year yield 0%
        [0.048, 0.072, 0.042, 0.416, 0.257, 0.110, 0.055],  # 30-year yield 1%
        [0.046, 0.076, 0.041, 0.401, 0.293, 0.096, 0.047],  # 30-year yield 2%
        [0.044, 0.083, 0.041, 0.380, 0.334, 0.080, 0.039],  # 30-year yield 3%
        [0.042, 0.091, 0.040, 0.353, 0.380, 0.064, 0.030],  # 30-year yield 4%
        [0.040, 0.100, 0.040, 0.310, 0.444, 0.046, 0.021],  # 30-year yield 5%
        [0.036, 0.107, 0.037, 0.256, 0.521, 0.030, 0.013],  # 30-year yield 6%
        [0.032, 0.106, 0.032, 0.211, 0.586, 0.022, 0.010],  # 30-year yield 7%
        [0.030, 0.103, 0.029, 0.143, 0.667, 0.020, 0.009],  # 30-year yield 8%
        [0.028, 0.100, 0.028, 0.068, 0.753, 0.016, 0.007],  # 30-year yield 9%
        [0.026, 0.100, 0.026, 0.026, 0.802, 0.014, 0.006],  # 30-year yield 10%
        [0.024, 0.093, 0.024, 0.020, 0.820, 0.013, 0.006],  # 30-year yield 11%
        [0.023, 0.088, 0.023, 0.019, 0.830, 0.012, 0.006],  # 30-year yield 12%
    ],
    [  # 5-year yield 12%
        [0.047, 0.065, 0.041, 0.428, 0.232, 0.123, 0.064],  # 30-year yield 0%
        [0.045, 0.068, 0.039, 0.419, 0.259, 0.113, 0.057],  # 30-year yield 1%
        [0.043, 0.072, 0.038, 0.404, 0.293, 0.100, 0.050],  # 30-year yield 2%
        [0.041, 0.077, 0.038, 0.386, 0.330, 0.087, 0.042],  # 30-year yield 3%
        [0.039, 0.082, 0.037, 0.365, 0.370, 0.072, 0.035],  # 30-year yield 4%
        [0.036, 0.087, 0.036, 0.334, 0.423, 0.057, 0.027],  # 30-year yield 5%
        [0.034, 0.092, 0.034, 0.287, 0.496, 0.040, 0.018],  # 30-year yield 6%
        [0.030, 0.092, 0.030, 0.241, 0.566, 0.028, 0.012],  # 30-year yield 7%
        [0.027, 0.089, 0.027, 0.200, 0.624, 0.023, 0.010],  # 30-year yield 8%
        [0.026, 0.087, 0.025, 0.124, 0.709, 0.020, 0.009],  # 30-year yield 9%
        [0.025, 0.087, 0.025, 0.061, 0.778, 0.016, 0.008],  # 30-year yield 10%
        [0.023, 0.088, 0.024, 0.027, 0.816, 0.014, 0.007],  # 30-year yield 11%
        [0.022, 0.084, 0.022, 0.022, 0.830, 0.013, 0.007],  # 30-year yield 12%
    ],
]


def look_up_portfolio(five_year_yield, thirty_year_yield):
    '''The portfolio for the given yields (in percent), for all scenarios at once.

    Yields that fall between two entries of the table get a mix of the portfolios around
    them: a yield of 3.25% gets 75% of the entry for 3% and 25% of the entry for 4%. This is
    done for both yields at the same time, so four entries are mixed. Yields below the
    lowest or above the highest entry get the portfolio of the nearest entry.
    '''
    table = torch.tensor(PORTFOLIOS, dtype=torch.float64)
    lowest, highest = TABLE_YIELDS[0], TABLE_YIELDS[-1]
    step = TABLE_YIELDS[1] - TABLE_YIELDS[0]

    # Where are we in the table? Position 3.25 means: a quarter of the way from entry 3 to entry 4.
    row_position = (five_year_yield.clamp(lowest, highest) - lowest) / step
    column_position = (thirty_year_yield.clamp(lowest, highest) - lowest) / step

    # The entry just below (the last entry has no next one, so stop one before the end) ...
    row = row_position.floor().long().clamp(max=len(TABLE_YIELDS) - 2)
    column = column_position.floor().long().clamp(max=len(TABLE_YIELDS) - 2)
    # ... and how far we are towards the next entry (0 = not at all, 1 = all the way).
    to_next_row = (row_position - row)[:, None]
    to_next_column = (column_position - column)[:, None]

    return ((1 - to_next_row) * (1 - to_next_column) * table[row, column]
            + to_next_row * (1 - to_next_column) * table[row + 1, column]
            + (1 - to_next_row) * to_next_column * table[row, column + 1]
            + to_next_row * to_next_column * table[row + 1, column + 1])


class Strategy:
    '''The strategy in the form that main.py of the case expects.'''

    def fit(self, train, cfg):
        pass  # nothing to learn: the table is fixed

    def act(self, obs):
        if obs.year == 0:
            return torch.tensor(FIRST_YEAR_PORTFOLIO, dtype=torch.float64), WITHDRAWAL_RATE
        end_of_last_year = obs.market_history[-1]
        five_year_yield = end_of_last_year[:, obs.market_names.index("US_ZC_5Y")]
        thirty_year_yield = end_of_last_year[:, obs.market_names.index("US_ZC_30Y")]
        return look_up_portfolio(five_year_yield, thirty_year_yield), WITHDRAWAL_RATE
