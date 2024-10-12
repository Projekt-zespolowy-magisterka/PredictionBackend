import numpy as np
from scipy.stats import ttest_rel
from tabulate import tabulate


def calculate_r_squared(y_test, y_pred):
    mean_y = np.mean(y_test)
    SST = np.sum((y_test - mean_y) ** 2)
    SSR = np.sum((y_test - y_pred) ** 2)
    rsquared = 1 - (SSR / SST)
    return rsquared


def pair_test(all_scores_table, models_names, metrics_names):
    n_experiment_objects = len(all_scores_table)
    n_metrics = len(metrics_names)

    t_student_matrix = np.zeros((n_experiment_objects, n_experiment_objects))
    p_matrix = np.zeros((n_experiment_objects, n_experiment_objects))
    better_metrics_matrix = np.zeros((n_experiment_objects, n_experiment_objects), dtype=bool)
    statistics_matters_matrix = np.zeros((n_experiment_objects, n_experiment_objects), dtype=bool)
    alpha = 0.05

    for metric_index in range(n_metrics):
        print(f"\n Test for metric: {metrics_names[metric_index]}")
        for i in range(n_experiment_objects):
            for j in range(n_experiment_objects):
                first_scores_table = all_scores_table[i, metric_index, :]
                # print (f" First scores table: {first_scores_table}")
                second_scores_table = all_scores_table[j, metric_index, :]
                # print(f" Second scores table: {second_scores_table}")
                stat, p_value = ttest_rel(first_scores_table, second_scores_table)

                t_student_matrix[i, j] = stat
                p_matrix[i, j] = p_value

                better_metrics_matrix[i, j] = np.mean(first_scores_table) > np.mean(second_scores_table)
                better_metrics_matrix[j, i] = np.mean(first_scores_table) <= np.mean(second_scores_table)
                statistics_matters_matrix[i, j] = p_value < alpha

        advantage_matter_stat_matrix = better_metrics_matrix * statistics_matters_matrix
        print("\n T-student matrix")
        print(tabulate(t_student_matrix, headers=models_names, showindex=models_names, tablefmt="grid"))
        print("\n P matrix")
        print(tabulate(p_matrix, headers=models_names, showindex=models_names, tablefmt="grid"))
        print("\n Better matrix")
        print(tabulate(better_metrics_matrix, headers=models_names, showindex=models_names, tablefmt="grid"))
        print("\n Stat matter matrix")
        print(tabulate(statistics_matters_matrix, headers=models_names, showindex=models_names, tablefmt="grid"))
        print("\n Adv matter matrix")
        print(tabulate(advantage_matter_stat_matrix, headers=models_names, showindex=models_names, tablefmt="grid"))
        print("\n")
