import os
from celery import Celery
from urllib.parse import urlparse, urlunparse

ENV = os.getenv("ENV", "PROD").upper()  # Default to PROD if not set

if ENV == "TEST":
    redis_broker = "redis://localhost:6379/0"
    redis_backend = "redis://localhost:6379/1"
elif ENV == "PROD":
    base_url = os.getenv("REDIS_URL")
    if not base_url:
        raise ValueError("In PROD environment, REDIS_URL must be set.")

    # Parse and switch DB index
    parsed = urlparse(base_url)
    redis_broker = urlunparse(parsed._replace(path="/0"))
    redis_backend = urlunparse(parsed._replace(path="/1"))
else:
    raise ValueError("ENV must be either 'TEST' or 'PROD'.")

celery_app = Celery(
    'rehal',
    broker=redis_broker,
    backend=redis_backend
)

celery_app.conf.update(
    result_expires=3600,  # Results expire after 1 hour
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    broker_transport_options={'visibility_timeout': 3600},  # Optional: Adjust as needed
)