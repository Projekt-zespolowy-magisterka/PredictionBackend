from flask import Blueprint, request, jsonify
from services.PredictionModelService import PredictionModelService
from services.ModelLearningService import ModelLearningService


prediction_controller_blueprint = Blueprint('prediction_controller', __name__)
model_service = PredictionModelService()
model_learn_service = ModelLearningService()


@prediction_controller_blueprint.route('/predict/<string:stock_symbol>', methods=['GET'])
def get_prediction(stock_symbol):
    try:
        interval, period, days_ahead = validate_request(stock_symbol)
        prediction = model_service.predict(stock_symbol, interval, period, days_ahead)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@prediction_controller_blueprint.route('/learn/<string:stock_symbol>', methods=['GET'])
def learn_models(stock_symbol):
    try:
        interval, period, days_ahead = validate_request(stock_symbol)
        model_learn_service.learn_models(stock_symbol, interval, period)
        return jsonify({'status': 'success', 'message': 'Models trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def validate_request(stock_symbol):
    interval = request.args.get('interval', default='1d')
    period = request.args.get('period', default='1y')
    days_ahead = request.args.get('days_ahead', default='1d')

    if not stock_symbol or not period or not interval:
        return jsonify({'error': 'Missing required parameters'}), 400

    valid_intervals = ['1m', '5m', '15m', '1h', '1d', '1wk', '1mo']
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    valid_days_ahead = ['1d', '2d', '3d']

    if interval not in valid_intervals:
        raise ValueError(f"[validate_request] Invalid interval: {interval}. Expected one of {valid_intervals}")
    if period not in valid_periods:
        raise ValueError(f"[validate_request] Invalid period: {period}. Expected one of {valid_periods}")
    if days_ahead not in valid_days_ahead:
        raise ValueError(f"[validate_request] Invalid days ahead for predict: {days_ahead}. Expected one of {valid_days_ahead}")

    return interval, period, days_ahead
