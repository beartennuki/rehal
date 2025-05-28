from celery import Celery
from dotenv import load_dotenv

load_dotenv()

celery_app = Celery(
    "rehal",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
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