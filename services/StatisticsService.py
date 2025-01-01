from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import pandas as pd
import os
import numpy as np
from app_config.ModelConfig import AVAILABLE_MODELS_NAMES, AVAILABLE_MODELS
from app_config.StatisticsConfig import metrics_array, metrics_names

matplotlib.use('Agg')


class StatisticsService:
    def __init__(self):
        self.models = AVAILABLE_MODELS
        self.models_names = AVAILABLE_MODELS_NAMES
        self.metrics_array = metrics_array
        self.metrics_names = metrics_names

    def create_stats_of_model(self, X_test, model, model_index, y_test, current_value_index, cv_scores, scaler_y):
        y_pred_scaled = model.predict(X_test)
        temp_model_min = np.min(y_pred_scaled, axis=0)
        temp_model_max = np.max(y_pred_scaled, axis=0)
        print(f"Model min: {temp_model_min}")
        print(f"Model max: {temp_model_max}")

        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, y_pred_scaled.shape[1]))
        aligned_y_test_original = scaler_y.inverse_transform(y_test)

        print(f"Y_pred {y_pred}")
        print(f"aligned_y_test_original {aligned_y_test_original}")

        print(f"y_pred_shape 0: {y_pred_scaled.shape[0]}")
        print(f"y_test_shape 0: {y_test.shape[0]}")
        print(f"X_test_shape 0: {X_test.shape[0]}")
        print(f"y_pred_shape 1: {y_pred_scaled.shape[1]}")
        print(f"y_test_shape 1: {y_test.shape[1]}")
        print(f"X_test_shape 1: {X_test.shape[1]}")

        for col_index in range(aligned_y_test_original.shape[1]):
            for metric_index, metric_function in enumerate(self.metrics_array):
                if metric_index != 2:
                    metric_value = metric_function(aligned_y_test_original[:, col_index], y_pred[:, col_index])
                    cv_scores[model_index, metric_index, current_value_index, col_index] = metric_value
                else:
                    metric_value = metric_function(y_test[:, col_index], y_pred_scaled[:, col_index])
                    cv_scores[model_index, metric_index, current_value_index, col_index] = metric_value
        return cv_scores, y_pred

    def create_stats_of_sequential_model(self, X_test, model, model_index, y_test, current_value_index, cv_scores, scaler_y):
        y_pred_scaled = model.predict(X_test)
        temp_model_min = np.min(y_pred_scaled, axis=0)
        temp_model_max = np.max(y_pred_scaled, axis=0)
        print(f"Model min: {temp_model_min}")
        print(f"Model max: {temp_model_max}")

        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, y_pred_scaled.shape[1]))
        aligned_y_test_original = scaler_y.inverse_transform(y_test)

        print(f"Y_pred {y_pred}")
        print(f"aligned_y_test_original {aligned_y_test_original}")

        print(f"[create_stats_of_sequential_model] y_pred_shape 0: {y_pred_scaled.shape[0]}")
        print(f"[create_stats_of_sequential_model] y_test_shape 0: {y_test.shape[0]}")
        print(f"[create_stats_of_sequential_model] X_test_shape 0: {X_test.shape[0]}")
        print(f"[create_stats_of_sequential_model] y_pred_shape 1: {y_pred_scaled.shape[1]}")
        print(f"[create_stats_of_sequential_model] y_test_shape 1: {y_test.shape[1]}")
        print(f"[create_stats_of_sequential_model] X_test_shape 1: {X_test.shape[1]}")

        for col_index in range(aligned_y_test_original.shape[1]):
            for metric_index, metric_function in enumerate(self.metrics_array):
                metric_value = metric_function(aligned_y_test_original[:, col_index], y_pred[:, col_index])
                cv_scores[model_index, metric_index, current_value_index, col_index] = metric_value
        return cv_scores, y_pred

    def save_stats_to_excel(self, X_test, X_train, current_model_name_key, current_value_index, model_index, stock_symbol, y_pred, y_test, y_train, cv_scores):
        folder_path = os.path.join("data_files", stock_symbol, "stats")
        os.makedirs(folder_path, exist_ok=True)

        results_df, train_results_df = self.create_dataframes_for_excel(X_test, X_train, y_pred, y_test, y_train)

        # TODO zmianić tutaj zeby bylo odwolanie do innego miejsca tutaj
        column_names = ["Open", "High", "Low", "Close", "Volume"]
        metrics_data = []
        for metric_index, metric_function in enumerate(self.metrics_array):
            for col_index in range(cv_scores.shape[3]):
                current_metric = cv_scores[model_index, metric_index, :, col_index]
                mean_value = np.mean(current_metric)
                std_value = np.std(current_metric)
                metrics_data.append({
                    'Metric': self.metrics_names[metric_index],
                    'Column': column_names[col_index],
                    'Mean': mean_value,
                    'Std': std_value
                })
                metrics_data.append({'Metric': '', 'Column': '', 'Mean': '', 'Std': ''})
                metrics_data.append({'Metric': '', 'Column': '', 'Mean': '', 'Std': ''})

        num_blank_columns = 4
        blank_columns = pd.DataFrame([[None] * num_blank_columns] * len(results_df), columns=[""] * num_blank_columns)
        metrics_df = pd.DataFrame(metrics_data)
        results_with_metrics_df = pd.concat([results_df, blank_columns, metrics_df], axis=1)
        fold_folder = os.path.join(folder_path, f"fold_{current_value_index}")
        os.makedirs(fold_folder, exist_ok=True)
        excel_file = os.path.join(fold_folder, f"model_results_{stock_symbol}_{current_model_name_key}_fold_{current_value_index}.xlsx")

        with pd.ExcelWriter(excel_file) as writer:
            results_with_metrics_df.to_excel(writer, sheet_name='Test Data', index=False)
            train_results_df.to_excel(writer, sheet_name='Train Data', index=False)
        return excel_file, fold_folder, results_df

    def display_results(self, cv_scores):
        for model_index, model in enumerate(self.models):
            print(f"Results for model: {self.models_names[model_index]}")
            for metric_index, metric_function in enumerate(self.metrics_array):
                print(f"  Metric: {self.metrics_names[metric_index]}")
                for col_index in range(cv_scores.shape[3]):
                    current_metric = cv_scores[model_index, metric_index, :, col_index]
                    mean_value = np.mean(current_metric)
                    std_value = np.std(current_metric)
                    print(f"    Column {col_index}: mean = {mean_value:.10f}, std = {std_value:.10f}")
            print()

    def create_dataframes_for_excel(self, X_test, X_train, y_pred, y_test, y_train):
        results_df = pd.DataFrame({
            'Day': X_test[:, 6],
            'Month': X_test[:, 7],
            'Year': X_test[:, 8],
            'Hour': X_test[:, 9],
            'y_test_Open': y_test[:, 0],
            'y_test_High': y_test[:, 1],
            'y_test_Low': y_test[:, 2],
            'y_test_Close': y_test[:, 3],
            'y_test_Volume': y_test[:, 4],
            'y_pred_Open': y_pred[:, 0],
            'y_pred_High': y_pred[:, 1],
            'y_pred_Low': y_pred[:, 2],
            'y_pred_Close': y_pred[:, 3],
            'y_pred_Volume': y_pred[:, 4],
        })
        train_results_df = pd.DataFrame({
            'Day_train': X_train[:, 6],
            'Month_train': X_train[:, 7],
            'Year_train': X_train[:, 8],
            'Hour_train': X_train[:, 9],
            'X_train_open': X_train[:, 0],
            'X_train_high': X_train[:, 1],
            'X_train_low': X_train[:, 2],
            'X_train_close': X_train[:, 3],
            'X_train_volume': X_train[:, 4],
            'X_train_return': X_train[:, 5],
            'y_train_open': y_train[:, 0],
            'y_train_high': y_train[:, 1],
            'y_train_low': y_train[:, 2],
            'y_train_close': y_train[:, 3],
            'y_train_volume': y_train[:, 4],
        })
        return results_df, train_results_df

    def save_chart_to_excel_file(self, current_model_name_key, current_value_index, excel_file, fold_folder, results_df,
                                 stock_symbol):
        image_folder = os.path.join(fold_folder, f"images")
        os.makedirs(image_folder, exist_ok=True)
        wb = load_workbook(excel_file)
        ws = wb.active
        current_row = len(results_df) + 5
        # TODO add here this enumaration diffrent source
        for feature_index, feature_name in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
            plt.figure(figsize=(20, 12))
            plt.plot(results_df['Day'].astype(str) + '-' + results_df['Month'].astype(str) + '-' + results_df['Year'].astype(str),
                results_df[f'y_test_{feature_name}'], label=f'Actual {feature_name}', color='blue', marker='o')
            plt.plot(results_df['Day'].astype(str) + '-' + results_df['Month'].astype(str) + '-' + results_df['Year'].astype(str),
                results_df[f'y_pred_{feature_name}'], label=f'Predicted {feature_name}', color='red', linestyle='--', marker='x')

            y_min, y_max = results_df[f'y_test_{feature_name}'].min(), results_df[f'y_test_{feature_name}'].max()
            tick_interval = (y_max - y_min) / 10
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))

            plt.xlabel('Date')
            plt.ylabel(f'{feature_name} Price')
            plt.title(f'Predicted vs Actual {feature_name} Price')
            plt.xticks(rotation=90)
            plt.legend()
            plt.tight_layout()

            chart_path = os.path.join(image_folder, f"chart_{feature_name}_{stock_symbol}_{current_model_name_key}_fold_{current_value_index}.png")
            plt.savefig(chart_path)
            plt.close()

            img = Image(chart_path)
            ws.add_image(img, f'A{current_row}')
            current_row += 80
        wb.save(excel_file)
        print(f"Results and charts saved to {excel_file}")

    def create_dataframes_for_excel_test(self, X_test, X_train, y_pred, y_test, y_train):
        results_df = pd.DataFrame({
            'Day': X_test[:, 1],
            'Month': X_test[:, 2],
            'Year': X_test[:, 3],
            'Hour': X_test[:, 4],
            'y_test_Open': y_test[:, 0],
            'y_test_High': y_test[:, 1],
            'y_test_Low': y_test[:, 2],
            'y_test_Close': y_test[:, 3],
            # 'y_test_Volume': y_test[:, 4],
            'y_pred_Open': y_pred[:, 0],
            'y_pred_High': y_pred[:, 1],
            'y_pred_Low': y_pred[:, 2],
            'y_pred_Close': y_pred[:, 3],
            # 'y_pred_Volume': y_pred[:, 4],
        })

        train_results_df = pd.DataFrame({
            'Day_train': X_train[:, 1],
            'Month_train': X_train[:, 2],
            'Year_train': X_train[:, 3],
            'Hour_train': X_train[:, 4],
            'y_train_open': y_train[:, 0],
            'y_train_high': y_train[:, 1],
            'y_train_low': y_train[:, 2],
            'y_train_close': y_train[:, 3],
            # 'y_train_volume': y_train[:, 4],
        })
        return results_df, train_results_df