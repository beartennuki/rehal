import os

from celery import Celery
from env import load_rehal_env

load_rehal_env()

redis_broker_url = os.getenv("REDIS_BROKER_URL")
if not redis_broker_url:
    raise EnvironmentError("REDIS_BROKER_URL is not set")

redis_result_backend = os.getenv("REDIS_RESULT_BACKEND")
if not redis_result_backend:
    raise EnvironmentError("REDIS_RESULT_BACKEND is not set")

celery_app = Celery(
    "rehal",
    broker=redis_broker_url,
    backend=redis_result_backend,
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Kuala_Lumpur',
    enable_utc=True,
    result_expires=3600,
    worker_log_format='[%(asctime)s:%(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s:%(levelname)s/%(processName)s] Task %(task_name)s[%(task_id)s]: %(message)s',
)

# import tasks so they register with this instance
import src.job.submit
import src.job.retrieve
import src.job.check_status
