from celery.exceptions import Ignore
from celery_app import celery_app
from src.submission import autoquiz, assessment

"""
Handle all submission status here.
This task must leave *exactly one* final SUCCESS/FAILURE record,
so the same code works with Redis, MongoDB, or any other backend.
"""

@celery_app.task(bind=True)
def submit_job(self, data):
    # ---------- basic validation ----------
    submit_info = data.get("submit_info")
    if not submit_info:
        err = {"status": "FAILED", "message": "Bad request – no submit_info"}
        self.update_state(state="FAILURE", meta=err)   # save details for clients
        raise Ignore()                                 # stop task without a second write

    submission_type = submit_info.get("submit_type")
    if submission_type == "autoquiz":
        respond = autoquiz.ATQ().start(submit_info)
    elif submission_type == "assessment":
        respond = assessment.Assessment().start(submit_info)
    elif submission_type == "reassessment":
        respond = assessment.Assessment().reassessment(submit_info)
    else:
        err = {"status": "FAILED", "message": "Unknown submission type"}
        self.update_state(state="FAILURE", meta=err)
        raise Ignore()

    # ---------- normal end-of-task handling ----------
    if respond.get("status") == "FAILED":
        # keep the full error dict in the backend
        self.update_state(state="FAILURE", meta=respond)
        raise Ignore()

    # SUCCESS path → just return the dictionary
    # Celery will write one final SUCCESS record that contains this dict.
    return respond