from repositories.FileModelRepository import FileModelRepository
from pymongo.errors import PyMongoError
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from utilities.dataReader import DataReader
from utilities.dataAnalyser import DataAnalyzer
from utilities.dataScaler import DataScaler
from utilities.statisticsCalculator import calculate_hurst_series
import yfinance as yf
import pandas as pd
import os
from flask import jsonify
import logging


class DataService:
    def __init__(self):
        self.file_repository = FileModelRepository()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.data_reader = DataReader()
        self.data_analyzer = DataAnalyzer()
        self.data_scaler = DataScaler()
        self.download_data = self.data_reader.get_download_data()
        self.upload_data = self.data_reader.get_upload_data()

    def get_stock_data_from_API(self, stock_symbol, interval, period):
        print("Starting data download")
        data = yf.download(stock_symbol, period=period, interval=interval, actions=True, prepost=True, threads=True)
        self.file_repository.save_to_parquet(data, stock_symbol, interval, period)
        print(data.head())
        print("Finished data download")

    def get_stock_ticker_data(self, stock_symbol, interval, period):
        # Define the stock symbol
        # stock_symbol = 'AAPL'  # Example: Apple Inc.
        ticker = yf.Ticker(stock_symbol)
        print("Ticker: \n")
        print(ticker)
        # List all available attributes and methods of the Ticker object
        print("\n \nTicker attributes\n")
        ticker_attributes = dir(ticker)
        print(ticker_attributes)
        # Fetch data
        # 1. Basic Information
        basic_info = ticker.info

        # 2. Historical Market Data
        historical_data = ticker.history(period=period, interval=interval, actions=True, prepost=True)

        # 3. Financials
        financials = ticker.financials
        quarterly_financials = ticker.quarterly_financials
        balance_sheet = ticker.balance_sheet
        quarterly_balance_sheet = ticker.quarterly_balance_sheet
        cashflow = ticker.cashflow
        quarterly_cashflow = ticker.quarterly_cashflow

        # # 4. Valuation Metrics
        # valuation = ticker.valuation

        # 5. Analyst Recommendations
        recommendations = ticker.recommendations

        # 6. Sustainability Data
        sustainability = ticker.sustainability

        # 7. Insider Transactions
        insider_transactions = ticker.insider_transactions

        # 8. Major Holders
        major_holders = ticker.major_holders

        # 9. Institutional Ownership
        institutional_holders = ticker.institutional_holders

        # 10. Options Data
        options = ticker.options  # List of expiration dates
        option_chain = ticker.option_chain(
            options[0]) if options else None  # Get option chain for the first expiration date

        # 11. Dividends and Splits
        dividends = ticker.dividends
        splits = ticker.splits

        # # 11 half. Shares
        # shares = ticker.shares

        # # 12. Current Quote
        # current_quote = ticker.quote

        # 13. News
        news = ticker.news

        # 14. Earnings
        # earnings = ticker.earnings depracated use:
        earnings = ticker.income_stmt

        quarterly_earnings = ticker.quarterly_earnings

        # Print retrieved data
        print("Basic Info:", basic_info)
        print("Historical Data:", historical_data.head())
        print("Financials:", financials)
        print("Quarterly Financials:", quarterly_financials)
        print("Balance Sheet:", balance_sheet)
        print("Quarterly Balance Sheet:", quarterly_balance_sheet)
        print("Cashflow:", cashflow)
        print("Quarterly Cashflow:", quarterly_cashflow)
        # print("Valuation:", valuation)
        print("Recommendations:", recommendations)
        print("Sustainability:", sustainability)
        print("Insider Transactions:", insider_transactions)
        print("Major Holders:", major_holders)
        print("Institutional Holders:", institutional_holders)
        print("Options Data:", options)
        print("Dividends:", dividends)
        print("Splits:", splits)
        # print("Current Quote:", current_quote)
        # print("Shares:", shares)

        print("News:", news)
        print("Earnings:", earnings)
        print("Quarterly Earnings:", quarterly_earnings)

        folder_path = stock_symbol
        os.makedirs(folder_path, exist_ok=True)

        # Save data to CSV or Parquet
        # Save historical market data to CSV
        historical_data.to_csv(os.path.join(folder_path, f"{stock_symbol}_historical_data.csv"))
        print(f"Historical data saved to {stock_symbol}_historical_data.csv")

        # Save financials to CSV
        financials.to_csv(os.path.join(folder_path, f"{stock_symbol}_financials.csv"))
        print(f"Financials data saved to {stock_symbol}_financials.csv")

        # Save quarterly financials to CSV
        quarterly_financials.to_csv(os.path.join(folder_path, f"{stock_symbol}_quarterly_financials.csv"))
        print(f"Quarterly financials data saved to {stock_symbol}_quarterly_financials.csv")

        # Save balance sheet to CSV
        balance_sheet.to_csv(os.path.join(folder_path, f"{stock_symbol}_balance_sheet.csv"))
        print(f"Balance sheet data saved to {stock_symbol}_balance_sheet.csv")

        # Save cashflow to CSV
        cashflow.to_csv(os.path.join(folder_path, f"{stock_symbol}_cashflow.csv"))
        print(f"Cashflow data saved to {stock_symbol}_cashflow.csv")

        # Optionally save other DataFrames if they are not empty
        if isinstance(dividends, pd.Series):
            dividends.to_csv(os.path.join(folder_path, f"{stock_symbol}_dividends.csv"))
            print(f"Dividends data saved to {stock_symbol}_dividends.csv")

        if isinstance(splits, pd.Series):
            splits.to_csv(os.path.join(folder_path, f"{stock_symbol}_splits.csv"))
            print(f"Splits data saved to {stock_symbol}_splits.csv")

        # Saving current quote to a single-row DataFrame and CSV
        # pd.DataFrame([current_quote]).to_csv(f"{stock_symbol}_current_quote.csv")
        # print(f"Current quote data saved to {stock_symbol}_current_quote.csv")
    def convert_parquet_to_csv(self, stock_symbol, interval, period):
        print("Started converting data")
        parquet_data = self.file_repository.load_parquet(stock_symbol, interval, period)
        self.file_repository.save_to_csv(parquet_data, stock_symbol, interval, period)
        print("Finished converting data")

    def analyze_data(self):
        self.data_analyzer.get_data_info()

    def get_parquet_data(self, stock_symbol, interval, period):
        return self.file_repository.load_parquet(stock_symbol, interval, period)

    def get_csv_data(self, stock_symbol, interval, period):
        return self.file_repository.load_csv(stock_symbol, interval, period)

    def process_data(self, data):
        print("[process_data] Processing of data started")
        # print(data)
        print("[process_data] Data columns:", data.columns)
        try:
            data['Return'] = data['Close'].pct_change()
            data.dropna(inplace=True)
            data['Year'] = data.index.year
            data['Month'] = data.index.month
            data['Day'] = data.index.day
            data['DayOfWeek'] = data.index.dayofweek
            data['IsWeekend'] = (data.index.dayofweek >= 5).astype(int)

            # data['Hurst'] = calculate_hurst_series(data['Close'])
            logging.error("Hurst finished")
            required_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek',
                                 'IsWeekend'
                # , 'Hurst'
                                 ]
            for feature in required_features:
                if feature not in data.columns:
                    error_message = f"Feature '{feature}' not found in data"
                    logging.error(error_message)
                    return jsonify({'error': error_message}), 400

            features = data[required_features]
            features = features.apply(pd.to_numeric, errors='coerce')
            print("[process_data] Feature data types:")
            print(features.dtypes)

            # target = data['Close']

            target = data[['Open', 'High', 'Low', 'Close']]

            # print("[process_data] FEATURES: ")
            # print(features)
            #
            # print("[process_data] TARGET: ")
            # print(target)
            target = target.dropna()

            # print("[process_data] NO NO NAN TARGET: ")
            # print(target)

            if len(features) != len(target):
                error_message = "Mismatch between features and target lengths after dropping NaNs"
                logging.error(error_message)
                return jsonify({'error': error_message}), 400

            print("[process_data] Processing finished")
            X = features
            y = target
            return X, y
        except Exception as e:
            print(jsonify({'error': str(e)}), 500)
            raise Exception(f"[process_data] Error: {e}")




