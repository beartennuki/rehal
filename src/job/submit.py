from celery_app import celery_app
from src.submission import autoquiz, assessment

'''
Handle all submission status here.
Update celery task and flask api here 
'''

@celery_app.task(bind=True)
def submit_job(self, data):

    submit_info = data.get('submit_info')
    if submit_info is None:
        msg = 'Bad Request, no load_info'
        self.update_state(state='FAILURE', meta={'message': msg})
        raise ValueError(msg)

    submission_type = submit_info.get('submit_type')
    if submission_type == 'autoquiz':
        respond = autoquiz.ATQ().start(submit_info)
    elif submission_type == 'assessment':
        respond = assessment.Assessment().start(submit_info)
    elif submission_type == 'reassessment':
        respond = assessment.Assessment().reassessment(submit_info)
    else:
        msg = 'Unknown submission type'
        self.update_state(state='FAILURE', meta={'message': msg})
        raise ValueError(msg)

    status = respond.get('status')
    msg = respond.get('message', 'Unknown failure')

    if status not in ['FAILED', 'SUCCESS']:
        raise ValueError('Unknown status in respond')

    if status == 'FAILED':
        self.update_state(state='SUCCESS', meta={'message': msg, 'error':True})
        return respond
    else:
        self.update_state(state='SUCCESS')
        return respond
