import numpy as np
from hurst import compute_Hc
import pandas as pd


def calculate_r_squared(y_test, y_pred):
    mean_y = np.mean(y_test)
    SST = np.sum((y_test - mean_y) ** 2)
    SSR = np.sum((y_test - y_pred) ** 2)
    rsquared = 1 - (SSR / SST)
    return rsquared


def calculate_hurst_series(data_close, old_data_close):
    hurst_values = []
    full_series = pd.concat([old_data_close, data_close])
    total_points = len(full_series)

    for i in range(len(old_data_close), total_points):
        window = full_series.iloc[i - 99:i + 1]
        H, c, _ = compute_Hc(window)
        hurst_values.append(round(H, 3))

    hurst_series = pd.Series(hurst_values, index=data_close.index)
    return hurst_series

