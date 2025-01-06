import os
import csv
from typing import List, Dict
import pandas as pd
import yfinance as yf

def _calculate_change(adj_close: pd.Series, days: int) -> float:
    """Calculate percentage change in adjusted close price over a specified period."""
    try:
        if len(adj_close) >= days:
            return round(((adj_close.iloc[-1] - adj_close.iloc[-days]) / adj_close.iloc[-days] * 100), 2)
    except Exception as e:
        print(f"Error calculating change over {days} days: {e}")
    return None


class StockService:
    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory
        os.makedirs(self.data_directory, exist_ok=True)

    def fetch_and_save_stock_data_from_validity(self, validity_file: str, filename: str = "stocks_data.csv") -> None:
        """Fetch stock data for all tickers in the validity column of a CSV file and save to a new CSV file."""
        if not os.path.exists(validity_file):
            raise FileNotFoundError(f"File {validity_file} does not exist.")

        tickers_df = pd.read_csv(validity_file)
        stock_symbols = tickers_df["ticker"].dropna().tolist()[:1000]

        data_list = []
        for stock_symbol in stock_symbols:
            try:
                data = yf.download(stock_symbol, period="5y", interval="1d", actions=True, prepost=True, threads=True)
                ticker = yf.Ticker(stock_symbol)
                stock_info = ticker.info

                adj_close = data['Adj Close']
                change_1m = _calculate_change(adj_close, 30)
                change_3m = _calculate_change(adj_close, 90)
                change_6m = _calculate_change(adj_close, 180)
                change_1y = _calculate_change(adj_close, 252)
                change_3y = _calculate_change(adj_close, 1095)

                name = stock_info.get("shortName", "N/A")
                if name != "N/A":
                    data_list.append({
                        "symbol": stock_symbol,
                        "name": name,
                        "price": adj_close.iloc[-1] if not adj_close.empty else None,
                        "peRatio": stock_info.get("trailingPE", "N/A"),
                        "volume": int(data['Volume'].iloc[-1]) if not data['Volume'].empty else None,
                        "change1M": change_1m,
                        "change3M": change_3m,
                        "change6M": change_6m,
                        "change1Y": change_1y,
                        "change3Y": change_3y
                    })

            except Exception as e:
                print(f"Error fetching data for {stock_symbol}: {e}")
                continue

        self._save_to_csv(data_list, filename)

    def load_stock_data_paginated(self, filename: str, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Load a paginated portion of stock data from the CSV file."""
        filepath = os.path.join(self.data_directory, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filename} does not exist.")

        chunk_size = 1000
        rows = []
        current_row = 0

        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            chunk_length = len(chunk)
            if current_row + chunk_length >= start_idx:
                rows.extend(chunk.iloc[max(0, start_idx - current_row): max(0, end_idx - current_row)].to_dict(
                    orient="records"))
                if len(rows) >= (end_idx - start_idx):
                    break
            current_row += chunk_length

        return pd.DataFrame(rows)

    def get_total_records(self, filename: str) -> int:
        """Get total number of records in the CSV file."""
        filepath = os.path.join(self.data_directory, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filename} does not exist.")

        with open(filepath, 'r') as file:
            return sum(1 for _ in file) - 1

    def get_score(self, symbol, ticker) -> pd.DataFrame:
        """Calculate scores from financial ratios."""
        def __normalize_and_score(row, ratio, min_val, max_val, weight):
            value = row.get(ratio)
            if value is None:
                return 0
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0, min(1, normalized))
            return normalized * weight
        
        def __calculate_allocation(score):
            min_val = 0.55
            max_val = 1
            buy_percentage = max(0, min(1, (score - min_val) / (max_val - min_val)))

            hold_percentage = max(0, min(1, (0.7 - abs(score - 0.55) * 2) / 0.7))

            min_val = 0
            max_val = 0.55
            sell_percentage = 1 - max(0, min(1, (score - min_val) / (max_val - min_val)))
            return buy_percentage * 100 , hold_percentage * 100, sell_percentage * 100

        try:
            info = ticker.info
            financial_ratios = {
                "Ticker": symbol,
                "P/E": info.get("trailingPE", None),
                "ROE": info.get("returnOnEquity", None),
                "Debt-to-Equity": info.get("debtToEquity", None),
                "Current Ratio": info.get("currentRatio", None),
            }
            df = pd.DataFrame([financial_ratios])

            weights = {
                "ROE": 0.3,
                "P/E": 0.3,
                "Current Ratio": 0.2,
                "Debt-to-Equity": 0.2,
            }

            benchmarks = {
                "ROE": {"min": 0.05, "max": 0.20},
                "P/E": {"min": 10, "max": 25},
                "Current Ratio": {"min": 1, "max": 3},
                "Debt-to-Equity": {"min": 0, "max": 2},
            }
            
            df["Score"] = df.apply(
                lambda row: sum(
                    __normalize_and_score(row, ratio, benchmarks[ratio]["min"], benchmarks[ratio]["max"], weights[ratio])
                    for ratio in weights.keys()
                ),
                axis=1
            )
            df[["buy", "hold", "sell"]] = df["Score"].apply(
                lambda score: pd.Series(__calculate_allocation(score))
            )
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return {}

    def get_stock_info(self, symbol: str) -> Dict:
        """Fetch and return stock data in the required JSON format."""
        try:
            ticker = yf.Ticker(symbol)
            stock_info = ticker.info
            print(f"Ticker: {stock_info}")

            data = yf.download(symbol, period="1y", interval="1h", actions=True, prepost=True, threads=True)

            if 'Adj Close' in data.columns:
                adj_close = data['Adj Close']
            elif 'Close' in data.columns:
                adj_close = data['Close']
            else:
                raise ValueError(f"Neither 'Adj Close' nor 'Close' data is available for symbol {symbol}")

            print(f"ADJ CLOSE (before conversion): {adj_close}")

            adj_close = adj_close.map(self.safe_get_value)
            adj_close = adj_close.squeeze()
            print(f"ADJ CLOSE (after conversion): {adj_close}")

            change_1m = _calculate_change(adj_close, 30)

            score_df = self.get_score(symbol, ticker)
            if isinstance(score_df, pd.DataFrame) and not score_df.empty:
                try:
                    score = score_df.iloc[0]
                    buy = float(score["buy"])
                    hold = float(score["hold"])
                    sell = float(score["sell"])
                except KeyError as e:
                    print(f"Key error in score_df: {e}")
                    buy, hold, sell = None, None, None
            else:
                buy, hold, sell = None, None, None

            chart_data = []
            for index, row in data.iterrows():
                chart_data.append({
                    "date": index.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": self.safe_get_value(row['Open']),
                    "high": self.safe_get_value(row['High']),
                    "low": self.safe_get_value(row['Low']),
                    "close": self.safe_get_value(row['Close']),
                    "volume": self.safe_get_value(row['Volume'])
                })

            stock_data = {
                "symbol": symbol,
                "name": stock_info.get("shortName", "N/A").item() if isinstance(stock_info.get("shortName", "N/A"), pd.Series) else stock_info.get("shortName", "N/A"),
                "currentPrice": float(adj_close.iloc[-1]) if not adj_close.empty else None,
                "change": {
                    "absolute": float(adj_close.iloc[-1] - adj_close.iloc[-30]) if len(adj_close) >= 30 else None,
                    "percentage": change_1m,
                },
                "scores": [
                    {"label": "Recommendation score", "buy": float(buy), "hold": float(hold), "sell": float(sell)},
                ],
                "statistics": [
                    {"label": "Market Cap",
                     "value": f"{stock_info.get('marketCap', 'N/A') / 1e12:.2f}T" if stock_info.get(
                         'marketCap') else "N/A"},
                    {"label": "P/E Ratio (TTM)",
                     "value": f"{stock_info.get('trailingPE', 'N/A')}" if stock_info.get('trailingPE') else "N/A"},
                    {"label": "EPS (TTM)",
                     "value": f"{stock_info.get('regularMarketEPS', 'N/A')} USD" if stock_info.get(
                         'regularMarketEPS') else "N/A"},
                    {"label": "Dividend Yield",
                     "value": f"{stock_info.get('dividendYield', 'N/A') * 100:.2f}%" if stock_info.get(
                         'dividendYield') else "N/A"},
                    {"label": "52-Week High",
                     "value": f"{stock_info.get('fiftyTwoWeekHigh', 'N/A')} USD" if stock_info.get(
                         'fiftyTwoWeekHigh') else "N/A"},
                    {"label": "52-Week Low",
                     "value": f"{stock_info.get('fiftyTwoWeekLow', 'N/A')} USD" if stock_info.get(
                         'fiftyTwoWeekLow') else "N/A"},
                ],
                "chartData": chart_data
            }
            return stock_data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return {}

    def _save_to_csv(self, data: List[Dict], filename: str) -> None:
        """Save stock data to a CSV file."""
        if not data:
            print("No data to save.")
            return

        filepath = os.path.join(self.data_directory, filename)
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def safe_get_value(self, value):
        if isinstance(value, pd.Series):
            return value.item()
        return value
