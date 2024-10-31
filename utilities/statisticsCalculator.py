import numpy as np
from scipy.stats import ttest_rel
from tabulate import tabulate
from hurst import compute_Hc
import pandas as pd


def calculate_r_squared(y_test, y_pred):
    mean_y = np.mean(y_test)
    SST = np.sum((y_test - mean_y) ** 2)
    SSR = np.sum((y_test - y_pred) ** 2)
    rsquared = 1 - (SSR / SST)
    return rsquared


def calculate_hurst_series(close_data):
    hurst_values = []
    print(f"Calculating hurst series for close_data: \n {close_data.head()}")
    print(f"Data size: {len(close_data)}")
    for i in range(len(close_data)):
        if i == 0:
            hurst_values.append(0.5)
        else:
            print(f"Slicing")
            closing_prices_slice = close_data.iloc[:i+1].values
            print(f"Closing price slices: \n {closing_prices_slice}")
            hurst = calculate_hurst(closing_prices_slice)
            print(f"Hurst value: {hurst}")
            hurst_values.append(hurst)
    print(f"Hurst values: \n {hurst_values}")
    return hurst_values


def calculate_hurst(closing_prices):
    print(f"DATA: {closing_prices}")
    if len(closing_prices) == 1:
        return 0.5

    H, c, data_series = compute_Hc(closing_prices, kind='price', simplified=True)
    print(f"Hurst exponent: {H}, c: {c}, time series: {data_series}")
    return H
