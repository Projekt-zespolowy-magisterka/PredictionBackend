import logging
import logging_loki

bind = "0.0.0.0:5000"
workers = 2
timeout = 360

# Logging configurations
loglevel = "info"  # Options: debug, info, warning, error, critical
accesslog = "-"  # Log access requests to stdout
errorlog = "-"  # Log errors to stdout
logger_class = "gunicorn.glogging.Logger"  # Use Gunicorn's default logger

# Customize Gunicorn logger to integrate with Python logging #TODO INFINITE LOOP
# def on_starting(server):
#     loki_url = "http://loki:3100/loki/api/v1/push"
#     handler = logging_loki.LokiHandler(
#         url=loki_url,
#         tags={"application": "prediction-mc"},
#         version="1"
#     )
#     handler.propagate = False
#
#     gunicorn_error_logger = logging.getLogger("gunicorn.error")
#     gunicorn_error_logger.addHandler(handler)
#     gunicorn_error_logger.setLevel(logging.DEBUG)
#
#     gunicorn_access_logger = logging.getLogger("gunicorn.access")
#     gunicorn_access_logger.addHandler(handler)
#     gunicorn_access_logger.setLevel(logging.DEBUG)
