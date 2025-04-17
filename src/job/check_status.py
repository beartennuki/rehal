from celery.result import AsyncResult
from celery_app import celery_app

def check_status(task_id):
    task = AsyncResult(task_id, app=celery_app)
    return {'state': task.state, 'info': task.info}