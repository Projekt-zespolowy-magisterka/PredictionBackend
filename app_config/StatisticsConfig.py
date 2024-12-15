from sklearn.metrics import mean_squared_error, mean_absolute_error
from utilities.statisticsCalculator import calculate_r_squared

metrics_array = [
    mean_squared_error,
    mean_absolute_error,
    calculate_r_squared
]

metrics_names = [
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared"
]
