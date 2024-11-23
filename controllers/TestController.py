from flask import Blueprint, request, jsonify

test_controller_blueprint = Blueprint('test_controller', __name__)


@test_controller_blueprint.route('/predictor/hello', methods=['GET'])
def hello():
    data = {'jack': 4098, 'sape': 4139, 'guido': 4127}
    try:
        return jsonify({'jsonData': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500