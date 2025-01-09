import os
import logging
from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
import logging_loki
from controllers.ModelController import prediction_controller_blueprint
from controllers.DataController import data_controller_blueprint
from controllers.TestController import test_controller_blueprint
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.sdk.resources import Resource
import tensorflow as tf

try:
    from tensorflow.python.distribute.cluster_resolver.tpu import TPUClusterResolver
    TPUClusterResolver = None  # Disable TPU detection explicitly
except ImportError:
    pass  # Ignore if TPU-related modules are not present

tf_logger = logging.getLogger("tensorflow")
tf_logger.propagate = False
tf.get_logger().setLevel(logging.ERROR)

loki_url = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

loki_handler = logging_loki.LokiHandler(
    url=loki_url,
    tags={"application": "prediction-mc"},
    version="1"
)
loki_handler.setLevel(logging.ERROR)
loki_handler.propagate = False

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

mainlogger = logging.getLogger()
mainlogger.setLevel(logging.ERROR)
mainlogger.addHandler(loki_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logger.addHandler(loki_handler)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)
matplotlib_logger.addHandler(loki_handler)

pillow_logger = logging.getLogger('PIL')
pillow_logger.setLevel(logging.WARNING)

mongologger = logging.getLogger('pymongo')
mongologger.setLevel(logging.ERROR)
mongologger.addHandler(loki_handler)

def on_starting(server):
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    gunicorn_error_logger.addHandler(loki_handler)
    gunicorn_error_logger.setLevel(logging.ERROR)

    gunicorn_access_logger = logging.getLogger("gunicorn.access")
    gunicorn_access_logger.addHandler(loki_handler)
    gunicorn_access_logger.setLevel(logging.ERROR)


app = Flask(__name__)
app.register_blueprint(prediction_controller_blueprint)
app.register_blueprint(data_controller_blueprint)
app.register_blueprint(test_controller_blueprint)
app.logger.setLevel(logging.ERROR)
app.logger.addHandler(loki_handler)

logger.info(f"Loki URL: {loki_url}")
logger.info(f"handler: {loki_handler}")

werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)
werkzeug_logger.addHandler(loki_handler)

metrics = PrometheusMetrics(app, path='/actuator/prometheus')

resource = Resource.create({"service.name": "prediction-mc"})
logger.info(f"resource: {resource}")
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer_provider = trace.get_tracer_provider()

zipkin_endpoint = os.getenv("ZIPKIN_ENDPOINT", "http://localhost:9411/api/v2/spans")
logger.info(f"Using Zipkin endpoint: {zipkin_endpoint}")
zipkin_exporter = ZipkinExporter(endpoint=zipkin_endpoint)
span_processor = BatchSpanProcessor(zipkin_exporter)
tracer_provider.add_span_processor(span_processor)

FlaskInstrumentor().instrument_app(app)


@app.route("/health")
def health_check():
    return {"status": "ok"}, 200


@app.before_request
def before_request():
    tracer = trace.get_tracer("prediction-mc")
    span = tracer.start_span(f"{request.method} {request.path}")
    span.set_attribute("http.method", request.method)
    span.set_attribute("http.url", request.url)
    request.span = span


@app.after_request
def after_request(response):
    if hasattr(request, "span"):
        request.span.set_attribute("http.status_code", response.status_code)
        request.span.end()
    return response


if __name__ == "__main__":
    logger.info("Starting Flask app with Loki logging enabled.", extra={"tags": {"service": "prediction-mc"}})
    app.run(debug=False)
