# celery_app.py
from celery import Celery

celery_app = Celery(
    'rehal',
    broker='amqp://guest:guest@localhost:5672//',
    backend='mongodb://localhost:27017/celery_beckend_db'
)

celery_app.conf.update(
    result_expires=3600,  # Results expire after 1 hour
) # this does not work, have to set manually TTL in mongosh
import src.job.submit