from flask import Blueprint, request, jsonify
from services.DataService import DataService

data_controller_blueprint = Blueprint('data_controller', __name__)
data_service = DataService()


@data_controller_blueprint.route('/data/<string:stock_symbol>', methods=['GET'])
def get_stock_data(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        data_service.get_stock_data_from_API(stock_symbol, interval, period)
        return jsonify({'status': 'success', 'message': 'Stock data got successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/data/convert/<string:stock_symbol>', methods=['POST'])
def convert_stock_parquet_to_csv(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        data_service.convert_parquet_to_csv(stock_symbol, interval, period)
        return jsonify({'status': 'success', 'message': 'Data converted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_controller_blueprint.route('/data/ticker/<string:stock_symbol>', methods=['GET'])
def get_stock_ticker_info(stock_symbol):
    try:
        interval, period = validate_request(stock_symbol)
        data_service.get_stock_ticker_data(stock_symbol, interval, period)
        return jsonify({'status': 'success', 'message': 'Data converted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def validate_request(stock_symbol):
    interval = request.args.get('interval', default='1d')
    period = request.args.get('period', default='1y')
    if not stock_symbol or not period or not interval:
        return jsonify({'error': 'Missing required parameters'}), 400

    valid_intervals = ['1m', '5m', '15m', '1h', '1d', '1wk', '1mo']
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval: {interval}. Expected one of {valid_intervals}")
    if period not in valid_periods:
        raise ValueError(f"Invalid period: {period}. Expected one of {valid_periods}")

    return interval, period
