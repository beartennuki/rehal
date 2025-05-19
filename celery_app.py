from celery import Celery
import os

ENV = os.getenv("REHAL_ENV_TYPE", "PROD").upper()

if ENV == "DEV":
    broker_url  = "redis://localhost:6379/0"   # queues
    backend_url = "redis://localhost:6379/1"   # results
elif ENV == "PROD":
    raw = os.environ["REDIS_URL"]              # e.g. redis://user:pass@host:6380
    # split broker/backend into two DBs
    broker_url  = raw.rstrip('/') + '/0'
    backend_url = raw.rstrip('/') + '/1'
else:
    raise ValueError("ENV must be either 'DEV' or 'PROD'.")

celery_app = Celery(
    "rehal",
    broker=broker_url,
    backend=backend_url,
)

celery_app.conf.update(
    result_expires=3600,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    broker_transport_options={"visibility_timeout": 3600},
    task_track_started=True,                   # nicer status reporting
)

# import tasks so they register with this instance
import src.job.submit
import src.job.retrieve
import src.job.check_status