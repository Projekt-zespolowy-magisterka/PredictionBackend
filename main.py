from flask import Flask, request, jsonify
from controllers.ModelController import prediction_controller_blueprint
from controllers.DataController import data_controller_blueprint
from controllers.TestController import test_controller_blueprint

app = Flask(__name__)
app.register_blueprint(prediction_controller_blueprint)
app.register_blueprint(data_controller_blueprint)
app.register_blueprint(test_controller_blueprint)

if __name__ == "__main__":
    app.run(debug=True)