import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from utilities.dataScaler import DataScaler


flag = True


# TODO zrobić coś z tymi nazwami ścieżek do zapisu
# TODO podpiąć ścieżke z file repository a tam dodać geta na bazowa sciezke
class DataAnalyzer:
    def __init__(self, output_dir="data_files"):
        self.data_scaler = DataScaler()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_data_info(self, data, stock_symbol):
        self.base_data_info(data, stock_symbol)
        self.analyze_data(data, stock_symbol)

    def analyze_data(self, data, stock_symbol):
        print()
        amount_of_rows_duplicated = data.duplicated().sum()
        print("Rows duplicated:")
        print(amount_of_rows_duplicated)
        print("\n")

        self.data_in_column_info(data, stock_symbol)
        self.plot_histograms(data, stock_symbol)
        self.plot_scatterplot_of_column(data, stock_symbol)
        self.plot_heat_map(data, stock_symbol)

    def data_in_column_info(self, data, stock_symbol):
        print("Amount of not unique throughput rows:")
        excluded_columns = ["Datetime", "Day", "Month", "Year", "Hour", "DayOfWeek", "IsWeekend"]
        for column_name in data.columns:
            if column_name not in excluded_columns:
                amount_of_not_unique_rows = data[column_name].nunique()
                print(f"Amount of not unique rows in {column_name}: {amount_of_not_unique_rows}")
                print("\n")
                if pd.api.types.is_numeric_dtype(data[column_name]):
                    mean = data[column_name].mean()
                    print(f"Mean value of {column_name}: {mean}")
                else:
                    print(f"{column_name} is not numeric. Skipping mean calculation.")
                print("\n")

    def base_data_info(self, data, stock_symbol):
        rows_amount = 100
        data_head = data.head(rows_amount)
        print()
        print(f'First {rows_amount} rows of dataset:')
        print(data_head)
        print("\n")

        print("Dataset datatypes:")
        print(data.dtypes)
        print("\n")

        print("Has dataset null values in some column:")
        print(data.isnull().any())
        print("\n")

    def plot_histograms(self, data, stock_symbol):
        excluded_columns = ["Datetime", "Day", "Month", "Year", "Hour", "DayOfWeek", "IsWeekend"]
        base_output_dir = os.path.join(self.output_dir, stock_symbol, "analysis", "histograms")
        os.makedirs(base_output_dir, exist_ok=True)

        for column in data.columns:
            if column not in excluded_columns:
                plt.figure()
                data[column].hist()
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {column}')
                output_path = os.path.join(base_output_dir, f"histogram_{column}.png")
                plt.savefig(output_path)
                plt.close()
                print(f"Histogram saved for {column} at {output_path}")

    def plot_scatterplot_of_column(self, data, stock_symbol):
        excluded_columns = ["Datetime", "Day", "Month", "Year", "Hour", "DayOfWeek", "IsWeekend"]
        base_output_dir = os.path.join(self.output_dir, stock_symbol, "analysis", "scatterplot")
        os.makedirs(base_output_dir, exist_ok=True)
        for column_name in data.columns:
            if column_name not in excluded_columns:
                plt.figure()
                data.plot(y=column_name, x=column_name, kind='scatter')
                plt.xlabel(column_name)
                plt.ylabel(column_name)
                plt.title(f'Scatterplot of {column_name}')
                output_path = os.path.join(base_output_dir, f"scatterplot_{column_name}.png")
                plt.savefig(output_path)
                plt.close()
                print(f"Scatterplot saved for {column_name} at {output_path}")

    def plot_heat_map(self, data, stock_symbol):
        numeric_data = data.select_dtypes(include=['number'])
        base_output_dir = os.path.join(self.output_dir, stock_symbol, "analysis", "heatmap")
        os.makedirs(base_output_dir, exist_ok=True)
        plt.figure(figsize=[20, 10])
        corr = numeric_data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.xticks(rotation=45)
        plt.title("Heatmap of Correlation Coefficient", size=12)
        output_path = os.path.join(base_output_dir, "heatmap.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Heatmap saved at {output_path}")
