import os
import pandas as pd


class FileModelRepository:
    CSV_FILE_EXTENSION = "csv"
    PARQUET_FILE_EXTENSION = "parquet"
    FILE_DIRECTORY = 'data_files'

    def __init__(self, base_dir=FILE_DIRECTORY):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def get_stock_dir(self, stock_symbol):
        stock_dir = os.path.join(self.base_dir, stock_symbol)
        os.makedirs(stock_dir, exist_ok=True)
        return stock_dir

    def save_to_csv(self, data, stock_symbol, interval, period):
        stock_dir = self.get_stock_dir(stock_symbol)
        file_path = os.path.join(stock_dir,
                                 self.generate_filename(stock_symbol, interval, period, self.CSV_FILE_EXTENSION))
        data.to_csv(file_path, index=True)
        print(f"Data saved to CSV at {file_path}")

    # TODO change saving methods here and how stock dir is got
    def save_to_parquet(self, data, stock_symbol, interval, period):
        stock_dir = self.get_stock_dir(stock_symbol)
        file_path = os.path.join(stock_dir,
                                 self.generate_filename(stock_symbol, interval, period, self.PARQUET_FILE_EXTENSION))
        data.to_parquet(file_path)
        print(f"Data saved to Parquet at {file_path}")

    def load_csv(self, stock_symbol, interval, period):
        stock_dir = self.get_stock_dir(stock_symbol)
        file_path = os.path.join(stock_dir, self.generate_filename(stock_symbol, interval, period, self.CSV_FILE_EXTENSION))
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
        try:
            data = pd.read_csv(file_path)
            print("Data loaded successfully.")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")

    def load_parquet(self, stock_symbol, interval, period):
        stock_dir = self.get_stock_dir(stock_symbol)
        file_path = os.path.join(stock_dir, self.generate_filename(stock_symbol, interval, period, self.PARQUET_FILE_EXTENSION))
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
        try:
            data = pd.read_parquet(file_path)
            print("Data loaded successfully.")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")

    def generate_filename(self, stock_symbol, interval, period, file_extension):
        return f"{stock_symbol}_interval_{interval}_period_{period}.{file_extension}"
