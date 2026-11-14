'''The yield model as a table: no neural network and no training needed.

THIS FILE WAS WRITTEN BY yield_mlp/yield_mlp.py. Do not change the numbers by hand; run
that file again to make a new table.

What it does: every year it looks at the 5-year and the 30-year US Treasury yield at the end
of last year and looks up the portfolio that belongs to them. The portfolios in the table
are the average of the portfolios proposed by 50 small neural networks.
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
FIRST_YEAR_PORTFOLIO = [0.171, 0.413, 0.178, 0.003, 0.232, 0.002, 0.002]

# The yields (in percent) for which the table has an entry. The same steps are used for the
# 5-year yield and for the 30-year yield.
TABLE_YIELDS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

# PORTFOLIOS[i][j] is the portfolio when the 5-year yield is TABLE_YIELDS[i] and the 30-year
# yield is TABLE_YIELDS[j]. Every portfolio is 7 weights in the order of ASSETS. Because of
# rounding they can add up to 0.999 or 1.001; the simulator scales them to exactly 1.
PORTFOLIOS = [
    [  # 5-year yield 0%
        [0.460, 0.275, 0.243, 0.003, 0.011, 0.003, 0.006],  # 30-year yield 0%
        [0.403, 0.329, 0.247, 0.001, 0.014, 0.002, 0.003],  # 30-year yield 1%
        [0.347, 0.371, 0.258, 0.001, 0.020, 0.001, 0.002],  # 30-year yield 2%
        [0.320, 0.393, 0.258, 0.001, 0.026, 0.001, 0.002],  # 30-year yield 3%
        [0.306, 0.406, 0.251, 0.001, 0.034, 0.001, 0.001],  # 30-year yield 4%
        [0.295, 0.418, 0.239, 0.001, 0.045, 0.001, 0.001],  # 30-year yield 5%
        [0.280, 0.431, 0.220, 0.001, 0.065, 0.001, 0.002],  # 30-year yield 6%
        [0.259, 0.440, 0.197, 0.002, 0.098, 0.002, 0.002],  # 30-year yield 7%
        [0.231, 0.437, 0.172, 0.002, 0.152, 0.003, 0.003],  # 30-year yield 8%
        [0.199, 0.420, 0.147, 0.003, 0.224, 0.003, 0.004],  # 30-year yield 9%
        [0.169, 0.394, 0.125, 0.004, 0.300, 0.004, 0.004],  # 30-year yield 10%
        [0.144, 0.364, 0.107, 0.005, 0.371, 0.005, 0.005],  # 30-year yield 11%
        [0.125, 0.334, 0.092, 0.006, 0.433, 0.006, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 1%
        [0.459, 0.256, 0.255, 0.004, 0.012, 0.006, 0.008],  # 30-year yield 0%
        [0.417, 0.312, 0.248, 0.002, 0.015, 0.002, 0.004],  # 30-year yield 1%
        [0.345, 0.370, 0.258, 0.001, 0.022, 0.001, 0.002],  # 30-year yield 2%
        [0.307, 0.401, 0.256, 0.001, 0.032, 0.001, 0.002],  # 30-year yield 3%
        [0.290, 0.417, 0.245, 0.001, 0.045, 0.001, 0.001],  # 30-year yield 4%
        [0.275, 0.431, 0.226, 0.001, 0.064, 0.001, 0.001],  # 30-year yield 5%
        [0.254, 0.445, 0.201, 0.001, 0.097, 0.001, 0.002],  # 30-year yield 6%
        [0.223, 0.444, 0.173, 0.002, 0.154, 0.002, 0.002],  # 30-year yield 7%
        [0.186, 0.424, 0.144, 0.003, 0.238, 0.003, 0.003],  # 30-year yield 8%
        [0.153, 0.389, 0.117, 0.004, 0.331, 0.004, 0.003],  # 30-year yield 9%
        [0.127, 0.350, 0.097, 0.004, 0.414, 0.005, 0.004],  # 30-year yield 10%
        [0.109, 0.314, 0.082, 0.005, 0.480, 0.005, 0.004],  # 30-year yield 11%
        [0.096, 0.284, 0.072, 0.006, 0.531, 0.006, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 2%
        [0.433, 0.237, 0.274, 0.012, 0.015, 0.013, 0.016],  # 30-year yield 0%
        [0.413, 0.293, 0.261, 0.004, 0.019, 0.004, 0.006],  # 30-year yield 1%
        [0.341, 0.364, 0.260, 0.002, 0.030, 0.002, 0.002],  # 30-year yield 2%
        [0.284, 0.411, 0.252, 0.001, 0.049, 0.001, 0.002],  # 30-year yield 3%
        [0.258, 0.433, 0.231, 0.001, 0.074, 0.001, 0.001],  # 30-year yield 4%
        [0.236, 0.447, 0.203, 0.001, 0.111, 0.001, 0.001],  # 30-year yield 5%
        [0.205, 0.448, 0.172, 0.002, 0.170, 0.002, 0.002],  # 30-year yield 6%
        [0.169, 0.423, 0.139, 0.003, 0.262, 0.003, 0.002],  # 30-year yield 7%
        [0.136, 0.378, 0.109, 0.003, 0.367, 0.003, 0.003],  # 30-year yield 8%
        [0.112, 0.331, 0.088, 0.004, 0.458, 0.004, 0.003],  # 30-year yield 9%
        [0.095, 0.292, 0.074, 0.005, 0.525, 0.005, 0.003],  # 30-year yield 10%
        [0.084, 0.262, 0.064, 0.006, 0.575, 0.005, 0.004],  # 30-year yield 11%
        [0.076, 0.239, 0.058, 0.007, 0.610, 0.006, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 3%
        [0.361, 0.216, 0.274, 0.066, 0.021, 0.032, 0.029],  # 30-year yield 0%
        [0.362, 0.271, 0.283, 0.027, 0.031, 0.014, 0.012],  # 30-year yield 1%
        [0.313, 0.347, 0.271, 0.006, 0.055, 0.004, 0.004],  # 30-year yield 2%
        [0.241, 0.413, 0.241, 0.002, 0.100, 0.002, 0.002],  # 30-year yield 3%
        [0.201, 0.436, 0.201, 0.002, 0.158, 0.002, 0.001],  # 30-year yield 4%
        [0.176, 0.432, 0.165, 0.002, 0.221, 0.002, 0.001],  # 30-year yield 5%
        [0.152, 0.411, 0.134, 0.003, 0.296, 0.002, 0.002],  # 30-year yield 6%
        [0.125, 0.372, 0.105, 0.004, 0.389, 0.003, 0.002],  # 30-year yield 7%
        [0.101, 0.323, 0.083, 0.004, 0.482, 0.004, 0.003],  # 30-year yield 8%
        [0.085, 0.280, 0.068, 0.005, 0.555, 0.005, 0.003],  # 30-year yield 9%
        [0.074, 0.247, 0.058, 0.006, 0.606, 0.005, 0.003],  # 30-year yield 10%
        [0.067, 0.223, 0.052, 0.007, 0.642, 0.006, 0.004],  # 30-year yield 11%
        [0.062, 0.206, 0.048, 0.007, 0.667, 0.006, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 4%
        [0.243, 0.190, 0.214, 0.233, 0.035, 0.050, 0.035],  # 30-year yield 0%
        [0.230, 0.228, 0.224, 0.209, 0.058, 0.032, 0.019],  # 30-year yield 1%
        [0.219, 0.291, 0.241, 0.122, 0.106, 0.014, 0.007],  # 30-year yield 2%
        [0.189, 0.366, 0.222, 0.020, 0.197, 0.004, 0.003],  # 30-year yield 3%
        [0.153, 0.392, 0.165, 0.004, 0.281, 0.003, 0.002],  # 30-year yield 4%
        [0.134, 0.392, 0.132, 0.003, 0.335, 0.003, 0.002],  # 30-year yield 5%
        [0.115, 0.372, 0.105, 0.004, 0.398, 0.003, 0.002],  # 30-year yield 6%
        [0.094, 0.332, 0.082, 0.005, 0.481, 0.004, 0.002],  # 30-year yield 7%
        [0.077, 0.283, 0.065, 0.005, 0.563, 0.005, 0.003],  # 30-year yield 8%
        [0.066, 0.242, 0.054, 0.006, 0.624, 0.005, 0.003],  # 30-year yield 9%
        [0.059, 0.214, 0.047, 0.007, 0.664, 0.006, 0.003],  # 30-year yield 10%
        [0.054, 0.194, 0.043, 0.007, 0.691, 0.006, 0.004],  # 30-year yield 11%
        [0.051, 0.181, 0.040, 0.008, 0.709, 0.007, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 5%
        [0.178, 0.171, 0.166, 0.332, 0.061, 0.056, 0.035],  # 30-year yield 0%
        [0.159, 0.198, 0.162, 0.325, 0.091, 0.042, 0.022],  # 30-year yield 1%
        [0.141, 0.231, 0.155, 0.292, 0.145, 0.025, 0.011],  # 30-year yield 2%
        [0.130, 0.271, 0.158, 0.186, 0.238, 0.011, 0.005],  # 30-year yield 3%
        [0.121, 0.332, 0.146, 0.026, 0.367, 0.005, 0.003],  # 30-year yield 4%
        [0.103, 0.346, 0.108, 0.006, 0.430, 0.004, 0.002],  # 30-year yield 5%
        [0.088, 0.332, 0.084, 0.006, 0.484, 0.005, 0.002],  # 30-year yield 6%
        [0.072, 0.295, 0.065, 0.006, 0.554, 0.005, 0.003],  # 30-year yield 7%
        [0.060, 0.249, 0.052, 0.007, 0.625, 0.005, 0.003],  # 30-year yield 8%
        [0.052, 0.212, 0.044, 0.007, 0.676, 0.006, 0.003],  # 30-year yield 9%
        [0.047, 0.188, 0.039, 0.008, 0.708, 0.006, 0.004],  # 30-year yield 10%
        [0.044, 0.172, 0.036, 0.008, 0.729, 0.007, 0.004],  # 30-year yield 11%
        [0.042, 0.161, 0.035, 0.009, 0.742, 0.007, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 6%
        [0.138, 0.157, 0.139, 0.367, 0.094, 0.066, 0.039],  # 30-year yield 0%
        [0.119, 0.175, 0.129, 0.361, 0.134, 0.054, 0.028],  # 30-year yield 1%
        [0.103, 0.194, 0.115, 0.343, 0.190, 0.038, 0.017],  # 30-year yield 2%
        [0.090, 0.211, 0.104, 0.294, 0.272, 0.020, 0.008],  # 30-year yield 3%
        [0.084, 0.241, 0.102, 0.180, 0.376, 0.011, 0.005],  # 30-year yield 4%
        [0.081, 0.289, 0.094, 0.024, 0.502, 0.007, 0.003],  # 30-year yield 5%
        [0.068, 0.288, 0.070, 0.009, 0.556, 0.006, 0.003],  # 30-year yield 6%
        [0.057, 0.259, 0.054, 0.008, 0.613, 0.006, 0.003],  # 30-year yield 7%
        [0.047, 0.219, 0.044, 0.008, 0.673, 0.006, 0.003],  # 30-year yield 8%
        [0.041, 0.188, 0.037, 0.008, 0.715, 0.007, 0.003],  # 30-year yield 9%
        [0.038, 0.167, 0.034, 0.009, 0.742, 0.007, 0.004],  # 30-year yield 10%
        [0.036, 0.154, 0.032, 0.009, 0.758, 0.007, 0.004],  # 30-year yield 11%
        [0.035, 0.146, 0.030, 0.010, 0.767, 0.008, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 7%
        [0.106, 0.139, 0.112, 0.393, 0.130, 0.078, 0.043],  # 30-year yield 0%
        [0.090, 0.150, 0.100, 0.389, 0.173, 0.065, 0.032],  # 30-year yield 1%
        [0.077, 0.160, 0.087, 0.377, 0.226, 0.050, 0.022],  # 30-year yield 2%
        [0.068, 0.170, 0.077, 0.344, 0.295, 0.033, 0.013],  # 30-year yield 3%
        [0.061, 0.184, 0.071, 0.279, 0.378, 0.019, 0.008],  # 30-year yield 4%
        [0.059, 0.213, 0.071, 0.146, 0.492, 0.012, 0.005],  # 30-year yield 5%
        [0.055, 0.242, 0.063, 0.022, 0.606, 0.008, 0.004],  # 30-year yield 6%
        [0.046, 0.225, 0.047, 0.010, 0.661, 0.007, 0.004],  # 30-year yield 7%
        [0.039, 0.193, 0.038, 0.009, 0.710, 0.007, 0.004],  # 30-year yield 8%
        [0.034, 0.167, 0.033, 0.009, 0.746, 0.007, 0.004],  # 30-year yield 9%
        [0.032, 0.150, 0.030, 0.010, 0.768, 0.008, 0.004],  # 30-year yield 10%
        [0.031, 0.139, 0.028, 0.010, 0.780, 0.008, 0.004],  # 30-year yield 11%
        [0.030, 0.133, 0.027, 0.010, 0.787, 0.008, 0.004],  # 30-year yield 12%
    ],
    [  # 5-year yield 8%
        [0.083, 0.123, 0.090, 0.413, 0.160, 0.086, 0.045],  # 30-year yield 0%
        [0.071, 0.129, 0.080, 0.409, 0.201, 0.074, 0.036],  # 30-year yield 1%
        [0.061, 0.135, 0.070, 0.399, 0.250, 0.059, 0.027],  # 30-year yield 2%
        [0.054, 0.140, 0.062, 0.375, 0.306, 0.044, 0.019],  # 30-year yield 3%
        [0.048, 0.150, 0.056, 0.328, 0.377, 0.029, 0.012],  # 30-year yield 4%
        [0.044, 0.161, 0.052, 0.263, 0.454, 0.019, 0.008],  # 30-year yield 5%
        [0.044, 0.187, 0.053, 0.099, 0.599, 0.013, 0.006],  # 30-year yield 6%
        [0.039, 0.192, 0.044, 0.020, 0.692, 0.009, 0.004],  # 30-year yield 7%
        [0.033, 0.170, 0.034, 0.012, 0.739, 0.008, 0.004],  # 30-year yield 8%
        [0.029, 0.148, 0.029, 0.011, 0.770, 0.008, 0.004],  # 30-year yield 9%
        [0.027, 0.135, 0.027, 0.011, 0.788, 0.008, 0.004],  # 30-year yield 10%
        [0.026, 0.126, 0.025, 0.011, 0.798, 0.008, 0.004],  # 30-year yield 11%
        [0.026, 0.121, 0.025, 0.011, 0.804, 0.009, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 9%
        [0.068, 0.110, 0.075, 0.427, 0.183, 0.092, 0.047],  # 30-year yield 0%
        [0.058, 0.114, 0.066, 0.422, 0.221, 0.080, 0.038],  # 30-year yield 1%
        [0.050, 0.116, 0.058, 0.411, 0.265, 0.068, 0.031],  # 30-year yield 2%
        [0.045, 0.120, 0.052, 0.393, 0.313, 0.054, 0.024],  # 30-year yield 3%
        [0.040, 0.125, 0.047, 0.361, 0.370, 0.040, 0.017],  # 30-year yield 4%
        [0.036, 0.132, 0.043, 0.311, 0.439, 0.027, 0.011],  # 30-year yield 5%
        [0.034, 0.140, 0.041, 0.229, 0.529, 0.019, 0.008],  # 30-year yield 6%
        [0.033, 0.155, 0.040, 0.070, 0.683, 0.013, 0.006],  # 30-year yield 7%
        [0.029, 0.148, 0.033, 0.020, 0.756, 0.010, 0.005],  # 30-year yield 8%
        [0.026, 0.133, 0.027, 0.013, 0.787, 0.009, 0.005],  # 30-year yield 9%
        [0.024, 0.121, 0.025, 0.012, 0.804, 0.009, 0.005],  # 30-year yield 10%
        [0.023, 0.115, 0.023, 0.012, 0.813, 0.009, 0.005],  # 30-year yield 11%
        [0.023, 0.111, 0.023, 0.012, 0.817, 0.009, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 10%
        [0.057, 0.099, 0.064, 0.436, 0.200, 0.096, 0.048],  # 30-year yield 0%
        [0.049, 0.101, 0.057, 0.430, 0.235, 0.086, 0.041],  # 30-year yield 1%
        [0.043, 0.103, 0.050, 0.420, 0.275, 0.074, 0.034],  # 30-year yield 2%
        [0.039, 0.105, 0.045, 0.404, 0.317, 0.062, 0.028],  # 30-year yield 3%
        [0.035, 0.108, 0.042, 0.380, 0.364, 0.050, 0.021],  # 30-year yield 4%
        [0.032, 0.113, 0.038, 0.343, 0.424, 0.036, 0.015],  # 30-year yield 5%
        [0.029, 0.115, 0.034, 0.294, 0.492, 0.026, 0.011],  # 30-year yield 6%
        [0.027, 0.122, 0.034, 0.168, 0.620, 0.020, 0.009],  # 30-year yield 7%
        [0.026, 0.126, 0.031, 0.054, 0.744, 0.014, 0.006],  # 30-year yield 8%
        [0.023, 0.119, 0.027, 0.020, 0.795, 0.011, 0.005],  # 30-year yield 9%
        [0.021, 0.111, 0.023, 0.015, 0.814, 0.010, 0.005],  # 30-year yield 10%
        [0.021, 0.105, 0.022, 0.014, 0.824, 0.010, 0.005],  # 30-year yield 11%
        [0.020, 0.102, 0.021, 0.013, 0.828, 0.010, 0.005],  # 30-year yield 12%
    ],
    [  # 5-year yield 11%
        [0.049, 0.090, 0.055, 0.444, 0.212, 0.100, 0.050],  # 30-year yield 0%
        [0.043, 0.091, 0.050, 0.437, 0.245, 0.091, 0.043],  # 30-year yield 1%
        [0.038, 0.092, 0.045, 0.426, 0.282, 0.080, 0.037],  # 30-year yield 2%
        [0.034, 0.094, 0.041, 0.412, 0.319, 0.069, 0.031],  # 30-year yield 3%
        [0.031, 0.096, 0.037, 0.393, 0.359, 0.058, 0.026],  # 30-year yield 4%
        [0.028, 0.098, 0.034, 0.365, 0.409, 0.045, 0.020],  # 30-year yield 5%
        [0.025, 0.099, 0.031, 0.325, 0.471, 0.033, 0.015],  # 30-year yield 6%
        [0.023, 0.099, 0.028, 0.265, 0.547, 0.026, 0.011],  # 30-year yield 7%
        [0.023, 0.105, 0.029, 0.122, 0.692, 0.020, 0.009],  # 30-year yield 8%
        [0.021, 0.105, 0.026, 0.046, 0.781, 0.014, 0.007],  # 30-year yield 9%
        [0.020, 0.102, 0.023, 0.020, 0.817, 0.012, 0.006],  # 30-year yield 10%
        [0.019, 0.097, 0.021, 0.016, 0.829, 0.011, 0.006],  # 30-year yield 11%
        [0.019, 0.094, 0.020, 0.015, 0.835, 0.011, 0.006],  # 30-year yield 12%
    ],
    [  # 5-year yield 12%
        [0.043, 0.083, 0.049, 0.449, 0.222, 0.103, 0.051],  # 30-year yield 0%
        [0.038, 0.083, 0.045, 0.441, 0.253, 0.095, 0.046],  # 30-year yield 1%
        [0.034, 0.084, 0.041, 0.430, 0.286, 0.085, 0.040],  # 30-year yield 2%
        [0.031, 0.086, 0.037, 0.417, 0.320, 0.075, 0.034],  # 30-year yield 3%
        [0.028, 0.087, 0.034, 0.402, 0.355, 0.065, 0.029],  # 30-year yield 4%
        [0.026, 0.088, 0.031, 0.381, 0.396, 0.054, 0.024],  # 30-year yield 5%
        [0.023, 0.088, 0.028, 0.348, 0.452, 0.042, 0.019],  # 30-year yield 6%
        [0.021, 0.086, 0.026, 0.307, 0.514, 0.033, 0.014],  # 30-year yield 7%
        [0.020, 0.088, 0.025, 0.213, 0.614, 0.027, 0.012],  # 30-year yield 8%
        [0.020, 0.092, 0.024, 0.095, 0.739, 0.020, 0.009],  # 30-year yield 9%
        [0.019, 0.092, 0.023, 0.041, 0.803, 0.015, 0.007],  # 30-year yield 10%
        [0.018, 0.091, 0.021, 0.021, 0.829, 0.013, 0.007],  # 30-year yield 11%
        [0.017, 0.089, 0.020, 0.018, 0.838, 0.012, 0.006],  # 30-year yield 12%
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
