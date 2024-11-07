import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utilities.dataScaler import DataScaler

flag = True

# TODO MORE REFACTOR TO SUITE FOR CURRENT IMPL
class DataAnalyzer:
    def __init__(self):
        self.data_scaler = DataScaler()
        self.base_data = True
        self.data_in_column = True
        self.speed_info = True
        self.plot_histograms = True
        self.plot_scatterplot = True
        self.plot_heat_map = True
        self.print_scaled_comparison = True

    def get_data_info(self, data):
        self.base_data_info(data)
        self.analyze_data(data)

    @staticmethod
    def analyze_data(data, current_data_type):
        print()
        amount_of_rows_duplicated = data.duplicated().sum()
        print("Rows duplicated:")
        print(amount_of_rows_duplicated)
        print("\n")

        # DataAnalyzer.data_in_column_info(data, column_name)
        DataAnalyzer.speed_info(data)
        # plot_histograms(data)
        # plot_scatterplot_of_column(data, column_name)
        DataAnalyzer.plot_heat_map(data)

    @staticmethod
    def data_in_column_info(data, column_name):
        print("Amount of not unique throughput rows:")
        amount_of_not_unique_rows = data[column_name].nunique()
        print(amount_of_not_unique_rows)
        print("\n")
        print("Mean throughput:")
        mean_throughput = data[column_name].mean()
        print(mean_throughput)
        print("\n")

    @staticmethod
    def base_data_info(data):
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

    @staticmethod
    def speed_info(data):
        speed_array = np.array(data['speed'])
        amount_of_non_zeroes = np.count_nonzero(speed_array)
        print("Amount of zeroes in speed column:")
        print(amount_of_non_zeroes)
        print("\n")

    @staticmethod
    def plot_histograms(data):
        for column in data.columns:
            data[column].hist()
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {column}')
            plt.show()

    @staticmethod
    def plot_scatterplot_of_column(data, column_name):
        for column in data.columns:
            data.plot(y=column_name, x=column, kind='scatter')
            plt.xlabel(column)
            plt.ylabel(column_name)
            plt.title(f'Scatterplot of {column}')
            plt.show()

    @staticmethod
    def plot_heat_map(data):
        corr = data.corr()
        plt.figure(figsize=[20, 10])
        sns.heatmap(corr, annot=True)
        plt.xticks(rotation=45)
        plt.title("Heatmap of Correlation Coefficient", size=12)
        plt.show()

    @staticmethod
    def plot_histograms_from_array(data, column_index):
        plt.hist(data)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of column with index {column_index}')
        plt.show()