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
        self.update_state(state='FAILURE', meta={'message': msg})
        raise ValueError(msg)

    doc_type = doc_info.get('doc_type')

    if doc_type == 'mcq':
        respond = MCQDoc().load_mcq(doc_info)
        if respond['status'] == 'FAILURE':
            err_msg = respond['message']
            self.update_state(state='FAILURE', meta={'message': err_msg})
            return {'status': 'FAILURE', 'message': err_msg}

        self.update_state(state='SUCCESS', meta={'message': 'Doc successfully retrieve'})
        doc = respond['doc']
        return {'status': 'SUCCESS', 'doc': doc}

    else:
        err_msg = 'Unknown doc type'
        self.update_state(state='FAILURE', meta={'message': err_msg})
        raise ValueError(err_msg)