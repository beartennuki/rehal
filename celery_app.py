from celery import Celery

celery_app = Celery(
    'rehal',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

celery_app.conf.update(
    result_expires=3600,  # Results expire after 1 hour
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    broker_transport_options={'visibility_timeout': 3600},  # Optional: Adjust as needed
)