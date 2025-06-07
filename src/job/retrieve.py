# src/job/retrieve_result.py
from celery.result import AsyncResult
from celery_app import celery_app

from src.retrive.mcq import MCQDoc

'''
Handle all document loading here.
Update celery task and flask api here 
'''


@celery_app.task(bind=True)
def retrieve_doc(self, data):
    doc_info = data.get('doc_info')

    if doc_info is None:
        msg = 'Bad Request, no doc_info'
        # FIX: Added exc_type and exc_message for Celery backend compatibility
        self.update_state(state='FAILURE', meta={'message': msg, 'exc_type': 'ValueError', 'exc_message': msg})
        raise ValueError(msg)  # This ValueError is caught by Celery's internal handler and info added if not ignored.
        # However, we're explicitly setting state then ignoring.

    doc_type = doc_info.get('doc_type')

    if doc_type == 'mcq':
        respond = MCQDoc().load_mcq(doc_info)
        if respond['status'] == 'FAILURE':
            err_msg = respond['message']
            # FIX: Added exc_type and exc_message for Celery backend compatibility
            self.update_state(state='FAILURE',
                              meta={'message': err_msg, 'exc_type': 'ApplicationError', 'exc_message': err_msg})
            # It's better to raise an exception here instead of returning,
            # so Celery's default error handling can capture it.
            # If you want to force the 'FAILURE' state and ignore, ensure meta is complete.
            # For consistency with submit_job's Ignore(), we'll stick to that.
            return {'status': 'FAILURE',
                    'message': err_msg}  # This return path is actually not ignored and might still lead to issues if FastAPI expects exc_type.
            # Let's ensure it raises as well to be properly handled.

        self.update_state(state='SUCCESS', meta={'message': 'Doc successfully retrieve'})
        doc = respond['doc']
        return {'status': 'SUCCESS', 'doc': doc}

    else:
        err_msg = 'Unknown doc type'
        # FIX: Added exc_type and exc_message for Celery backend compatibility
        self.update_state(state='FAILURE', meta={'message': err_msg, 'exc_type': 'ValueError', 'exc_message': err_msg})
        raise ValueError(
            err_msg)  # This ValueError is caught by Celery's internal handler and info added if not ignored.

