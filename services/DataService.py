from repositories.FileModelRepository import FileModelRepository
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from utilities.dataAnalyser import DataAnalyzer
from utilities.statisticsCalculator import calculate_hurst_series
import yfinance as yf
import pandas as pd
import os
from flask import jsonify, current_app
import logging
import json


class DataService:
    def __init__(self):
        self.file_repository = FileModelRepository()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.data_analyzer = DataAnalyzer()

    def get_stock_data_from_API(self, stock_symbol, interval, period):
        current_app.logger.info("Starting data download")
        data = yf.download(stock_symbol, period=period, interval=interval, actions=True, prepost=True, threads=True)
        ticker = yf.Ticker(stock_symbol)
        stock_info = ticker.info
        old_data = yf.download(stock_symbol, period='5y', interval='1d', actions=True, prepost=True, threads=True)

        adj_close = data['Adj Close']
        change_1m = ((adj_close[-1] - adj_close[-30]) / adj_close[-30] * 100) if len(adj_close) >= 30 else None
        change_3m = ((adj_close[-1] - adj_close[-90]) / adj_close[-90] * 100) if len(adj_close) >= 90 else None
        change_6m = ((adj_close[-1] - adj_close[-180]) / adj_close[-180] * 100) if len(adj_close) >= 180 else None
        change_1y = ((adj_close[-1] - adj_close[-252]) / adj_close[-252] * 100) if len(adj_close) >= 252 else None

        print(data.head())

        response = {
            "symbol": stock_symbol,
            "name": stock_info.get("shortName", "N/A"),
            "price": adj_close[-1],
            "peRatio": stock_info.get("trailingPE", "N/A"),
            "volume": int(data['Volume'].iloc[-1]),
            "change1M": round(change_1m, 2) if change_1m is not None else None,
            "change3M": round(change_3m, 2) if change_3m is not None else None,
            "change6M": round(change_6m, 2) if change_6m is not None else None,
            "change1Y": round(change_1y, 2) if change_1y is not None else None
        }
        processed_data = self.process_data(data, old_data)
        self.file_repository.save_to_parquet(processed_data, stock_symbol, interval, period)
        self.file_repository.save_to_csv(processed_data, stock_symbol, interval, period)
        print("Finished data download")
        return [response]

    def get_stock_chart_data_from_API(self, stock_symbol, interval, period):
        current_app.logger.info("Starting data download")
        data = yf.download(stock_symbol, period=period, interval=interval, actions=True, prepost=True, threads=True)

        print(data.head())

        response = [
            {
                "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "volume": row["Volume"]
            }
            for date, row in data.iterrows()
        ]
        response_df = pd.DataFrame(response)

        # TODO change this beacause it create unnececary chart directory but except that work ok
        base_output_dir = os.path.join("chart")
        os.makedirs(base_output_dir, exist_ok=True)
        self.file_repository.save_to_parquet(response_df, base_output_dir, interval, period)
        self.file_repository.save_to_csv(response_df, base_output_dir, interval, period)
        print("Finished data download")
        return [response]

    def convert_parquet_to_csv(self, stock_symbol, interval, period):
        print("Started converting data")
        parquet_data = self.file_repository.load_parquet(stock_symbol, interval, period)
        self.file_repository.save_to_csv(parquet_data, stock_symbol, interval, period)
        print("Finished converting data")

    def save_predictions_to_csv(self, data, stock_symbol, interval, period):
        print("Started converting data")
        pred_stock_name = stock_symbol + "_PRED"
        self.file_repository.save_to_csv(data, pred_stock_name, interval, period)
        print("Finished converting data")

    def analyze_data(self, stock_symbol, interval, period):
        data = self.file_repository.load_csv(stock_symbol, interval, period)
        self.data_analyzer.get_data_info(data, stock_symbol)

    def get_parquet_data(self, stock_symbol, interval, period):
        return self.file_repository.load_parquet(stock_symbol, interval, period)

    def get_csv_data(self, stock_symbol, interval, period):
        return self.file_repository.load_csv(stock_symbol, interval, period)

    def process_data(self, data, old_data):
        print("[process_data] Processing of data started")
        print("[process_data] Data columns:", data.columns)
        try:
            # TODO change timestamp from now to last from dataset
            three_years_ago = pd.Timestamp.now() - pd.DateOffset(years=3)
            old_data_filtered = old_data[old_data.index >= three_years_ago].tail(100)
            print(f"old data: \n {old_data_filtered}")
            hurst_series = calculate_hurst_series(data['Close'], old_data_filtered['Close'])
            data.drop(columns=['Adj Close', 'Dividends', 'Stock Splits'], inplace=True)  #TODO sprawdzic dzialanie aktualne z dywidendami i spliatami
            data['Return'] = data['Close'].pct_change()
            data.dropna(inplace=True)
            data['Day'] = data.index.day
            data['Month'] = data.index.month
            data['Year'] = data.index.year
            data['Hour'] = data.index.hour
            data['DayOfWeek'] = data.index.dayofweek
            data['IsWeekend'] = (data.index.dayofweek >= 5).astype(int)
            data['Hurst'] = hurst_series
            data['MA_10'] = data['Close'].rolling(window=10).mean() #TODO dorobić tutaj poprawne wpisyanie w poczatkowych rekordach
            data['MA_50'] = data['Close'].rolling(window=50).mean()

            print(f"data head: \n {data.head()}")
            return data
        except Exception as e:
            print(jsonify({'error': str(e)}), 500)
            raise Exception(f"[process_data] Error: {e}")

    # TODO ADD MINUTES
    # TODO Make two get objectives with no values in X_test and with values in X_test that match values I want to predict
    # TODO Make same two X wihout close, high etc but with shift (shift(-1))?
    # TODO add random zeros in first scenario to close, open, etc?
    def get_objectives_from_data(self, processed_data):
        try:
            # TODO tutaj brakuje ME_10 i ME_50 do podpiedzia
            # TODO wrzucic te odwolanie do innego miejsca
            required_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Day', 'Month', 'Year', 'Hour', 'DayOfWeek', 'IsWeekend', 'Hurst']
            for feature in required_features:
                if feature not in processed_data.columns:
                    error_message = f"Feature '{feature}' not found in data"
                    logging.error(error_message)
                    return jsonify({'error': error_message}), 400

            features = processed_data[required_features]
            features = features.apply(pd.to_numeric, errors='coerce')
            print("[get_objectives_from_data] Feature data types:")
            print(features.dtypes)

            # TODO wrzucic to do innego miejsca to odwolanie
            target = processed_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            target = target.dropna()

            if len(features) != len(target):
                error_message = "Mismatch between features and target lengths after dropping NaNs"
                logging.error(error_message)
                return jsonify({'error': error_message}), 400

            print("[get_objectives_from_data] Getting objectives from data finished")
            X = features.copy()
            y = target.copy()
            return X, y
        except Exception as e:
            print(jsonify({'error': str(e)}), 500)
            raise Exception(f"[get_objectives_from_data] Error: {e}")

    # Ticker info just for analyze
    def get_stock_ticker_data(self, stock_symbol, interval, period):
        ticker = yf.Ticker(stock_symbol)
        print("Ticker: \n", ticker)

        print("\n\nTicker attributes\n")
        ticker_attributes = dir(ticker)
        print(ticker_attributes)

        try:
            # 1. Basic Information
            basic_info = ticker.info

            # 2. Historical Market Data
            historical_data = ticker.history(period=period, interval=interval, actions=True, prepost=True)

            # Helper function to safely convert data to JSON-serializable format
            def safe_to_dict(obj):
                if isinstance(obj, pd.DataFrame):
                    # Reset index and convert keys to strings
                    obj = obj.reset_index()
                    obj.columns = [str(col) if not isinstance(col, str) else col for col in obj.columns]
                    return obj.applymap(
                        lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x
                    ).to_dict(orient="records")
                if isinstance(obj, pd.Series):
                    return obj.apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x).to_dict()
                if isinstance(obj, dict):
                    return {str(k): v for k, v in obj.items()}  # Convert keys to strings
                return {}

            # Financial data
            financials = safe_to_dict(ticker.financials)
            quarterly_financials = safe_to_dict(ticker.quarterly_financials)
            balance_sheet = safe_to_dict(ticker.balance_sheet)
            quarterly_balance_sheet = safe_to_dict(ticker.quarterly_balance_sheet)
            cashflow = safe_to_dict(ticker.cashflow)
            quarterly_cashflow = safe_to_dict(ticker.quarterly_cashflow)

            # Recommendations and other data
            recommendations = ticker.recommendations.reset_index().apply(
                lambda x: x.map(lambda y: y.isoformat() if isinstance(y, pd.Timestamp) else y)
            ).to_dict(orient="records") if isinstance(ticker.recommendations, pd.DataFrame) else []

            sustainability = safe_to_dict(ticker.sustainability)
            insider_transactions = ticker.insider_transactions.reset_index().apply(
                lambda x: x.map(lambda y: y.isoformat() if isinstance(y, pd.Timestamp) else y)
            ).to_dict(orient="records") if isinstance(ticker.insider_transactions, pd.DataFrame) else []

            major_holders = safe_to_dict(ticker.major_holders)
            institutional_holders = safe_to_dict(ticker.institutional_holders)

            # Options Data
            options = list(ticker.options) if ticker.options else []

            # Dividends and Splits
            dividends = safe_to_dict(ticker.dividends)
            splits = safe_to_dict(ticker.splits)

            # News
            news = ticker.news

            # Earnings
            earnings = safe_to_dict(ticker.income_stmt)
            quarterly_earnings = safe_to_dict(ticker.quarterly_earnings)

            # Construct the result dictionary
            result = {
                "basic_info": basic_info,
                "historical_data": safe_to_dict(historical_data),
                "financials": financials,
                "quarterly_financials": quarterly_financials,
                "balance_sheet": balance_sheet,
                "quarterly_balance_sheet": quarterly_balance_sheet,
                "cashflow": cashflow,
                "quarterly_cashflow": quarterly_cashflow,
                "recommendations": recommendations,
                "sustainability": sustainability,
                "insider_transactions": insider_transactions,
                "major_holders": major_holders,
                "institutional_holders": institutional_holders,
                "options": options,
                "news": news,
                "earnings": earnings,
                "quarterly_earnings": quarterly_earnings,
            }

            # Convert the result to JSON
            response_json = json.dumps(result, indent=4)

            # Save data to folder
            # TODO zrobić tutaj ładniej to z tymi scieżkami
            base_output_dir = os.path.join("data_files", stock_symbol, "ticker")
            os.makedirs(base_output_dir, exist_ok=True)

            # Save historical data
            historical_data.to_csv(os.path.join(base_output_dir, f"{stock_symbol}_historical_data.csv"))

            # Save other datasets
            for name, data in [
                ("financials", financials),
                ("quarterly_financials", quarterly_financials),
                ("balance_sheet", balance_sheet),
                ("cashflow", cashflow),
            ]:
                if isinstance(data, list) and data:
                    pd.DataFrame(data).to_csv(os.path.join(base_output_dir, f"{stock_symbol}_{name}.csv"))

            if isinstance(dividends, dict):
                pd.Series(dividends).to_csv(os.path.join(base_output_dir, f"{stock_symbol}_dividends.csv"))

            if isinstance(splits, dict):
                pd.Series(splits).to_csv(os.path.join(base_output_dir, f"{stock_symbol}_splits.csv"))

            return response_json

        except Exception as e:
            print(f"Error processing stock data: {e}")
            return {"error": str(e)}