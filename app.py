from flask import Flask, request, jsonify
from controllers.ModelController import prediction_controller_blueprint
from controllers.DataController import data_controller_blueprint
from controllers.TestController import test_controller_blueprint
from prometheus_flask_exporter import PrometheusMetrics
import logging_loki
import logging

app = Flask(__name__)
app.register_blueprint(prediction_controller_blueprint)
app.register_blueprint(data_controller_blueprint)
app.register_blueprint(test_controller_blueprint)

# loki_url = 'http://localhost:3100/loki/api/v1/push'

#TODO DOCKER LOKI SETUP USE PROFILES
loki_url = 'http://loki:3100/loki/api/v1/push'


handler = logging_loki.LokiHandler(
    url=loki_url,
    tags={"application": "prediction-mc"},
    version="1"
)

app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.addHandler(handler)
werkzeug_logger.setLevel(logging.DEBUG)

metrics = PrometheusMetrics(app)

if __name__ == "__main__":
    app.logger.info("Starting Flask app with Loki logging enabled.", extra={"tags": {"service": "prediction-mc"}})
    app.run(debug=False)