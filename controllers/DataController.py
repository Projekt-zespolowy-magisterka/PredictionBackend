import datetime
import os

from flask import Blueprint, request, jsonify
from services.DataService import DataService
import pandas as pd

from services.StockService import StockService

data_controller_blueprint = Blueprint('data_controller', __name__)
data_service = DataService()

VALID_INTERVALS = ['1m', '5m', '15m', '1h', '1d', '1wk', '1mo']
VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
CSV_FILE_PATH = "data_files/valid_tickers.csv"


def validate_request(stock_symbol):
    interval = request.args.get('interval', default='1d')
    period = request.args.get('period', default='1y')

    if not stock_symbol or not period or not interval:
        raise ValueError('Missing required parameters')

    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval: {interval}. Expected one of {VALID_INTERVALS}")
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period: {period}. Expected one of {VALID_PERIODS}")

    return interval, period


# Route handlers
@data_controller_blueprint.route('/predictor/data/<string:stock_symbol>', methods=['GET'])
def get_stock_data(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        response = data_service.get_stock_data_from_API(stock_symbol, interval, period)
        return jsonify({'status': 'success', 'data': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/predictor/data/chart/<string:stock_symbol>', methods=['GET'])
def get_stock_chart_data(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        response = data_service.get_stock_chart_data_from_API(stock_symbol, interval, period)
        return jsonify({'status': 'success', 'data': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/predictor/data/saved/<string:stock_symbol>', methods=['GET'])
def get_saved_stock_data(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        data = data_service.get_parquet_data(stock_symbol, interval, period)
        data_json = data.to_dict(orient='records')
        return jsonify({'status': 'success', 'message': 'Stock data retrieved successfully', 'data': data_json})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/predictor/data/prediction/<string:stock_symbol>', methods=['GET'])
def get_stock_prediction(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        prediction_path = f"{stock_symbol}_PRED"
        data = data_service.get_csv_data(prediction_path, interval, period)
        data_json = data.to_dict(orient='records')
        return jsonify({'status': 'success', 'message': 'Prediction data retrieved successfully', 'data': data_json})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/predictor/data/convert/<string:stock_symbol>', methods=['GET'])
def convert_stock_parquet_to_csv(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        data_service.convert_parquet_to_csv(stock_symbol, interval, period)
        return jsonify({'status': 'success', 'message': 'Data converted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/predictor/data/analyze/<string:stock_symbol>', methods=['GET'])
def get_data_analyze(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        data_service.analyze_data(stock_symbol, interval, period)

        return jsonify({'status': 'success', 'message': 'Analysis completed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/predictor/data/ticker/<string:stock_symbol>', methods=['GET'])
def get_stock_ticker(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        ticker = data_service.get_stock_ticker_data(stock_symbol, interval, period)
        data_json = ticker.to_dict(orient='records')
        return jsonify({'status': 'success', 'message': 'Ticker retrieved successfully', 'data': data_json})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/predictor/data/tickers', methods=['GET'])
def get_all_stock_tickers():
    try:
        # Pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        stock_service = StockService()
        filename = "stocks_data.csv"

        # Calculate pagination
        total_records = stock_service.get_total_records(filename)
        total_pages = (total_records + per_page - 1) // per_page

        if page < 1 or page > total_pages:
            raise ValueError(f"Page {page} is out of range. Total pages: {total_pages}")

        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_records)

        # Load paginated data
        paginated_data = stock_service.load_stock_data_paginated(filename, start_idx, end_idx)

        return jsonify({
            "status": "success",
            "data": paginated_data.to_dict(orient='records'),
            "pagination": {
                "current_page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "total_records": total_records
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/predictor/data/save', methods=['POST'])
def save_stock_data():
    try:
        # Fetch and save stock data to CSV
        stock_service = StockService()
        stock_service.fetch_and_save_stock_data_from_validity("data_files/valid_new_tickers.csv", "stocks_data.csv")

        return jsonify({"status": "success", "message": "Stock data saved successfully."})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/predictor/data/stock_info/<string:stock_symbol>', methods=['GET'])
def get_stock_info(stock_symbol):
    try:
        stock_service = StockService()
        stock_data = stock_service.get_stock_info(stock_symbol)

        if not stock_data:
            return jsonify({'error': f"No data found for symbol {stock_symbol}"}), 404

        return jsonify({'status': 'success', 'data': stock_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500