from flask import Blueprint, request, jsonify
from services.PredictionModelService import PredictionModelService
from services.ModelLearningService import ModelLearningService
import pandas as pd
import os

prediction_controller_blueprint = Blueprint('prediction_controller', __name__)
model_service = PredictionModelService()
model_learn_service = ModelLearningService()


@prediction_controller_blueprint.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        prediction = model_service.predict(data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@prediction_controller_blueprint.route('/hello', methods=['GET'])
def hello():
    data = {'jack': 4098, 'sape': 4139, 'guido': 4127}
    try:
        return jsonify({'jsonData': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@prediction_controller_blueprint.route('/learn-test', methods=['GET'])
def learn_models_test():
    try:
        model_learn_service.learn_models_test()
        return jsonify({'status': 'success', 'message': 'Models trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@prediction_controller_blueprint.route('/learn', methods=['GET'])
def learn_models():
    try:
        model_learn_service.learn_models()
        return jsonify({'status': 'success', 'message': 'Models trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
