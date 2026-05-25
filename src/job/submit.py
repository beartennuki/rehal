from celery.utils.log import get_task_logger
from celery.exceptions import Ignore
from celery_app import celery_app
from src.submission import article_generation, autoquiz, assessment, canonical_topic

logger = get_task_logger(__name__)

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
        err_msg = "Bad request – no submit_info"
        # FIX: Added exc_type and exc_message for Celery backend compatibility
        err_meta = {"status": "FAILED", "message": err_msg, "exc_type": "ValueError", "exc_message": err_msg}
        logger.error("submit_job failed before dispatch: %s", err_msg)
        self.update_state(state="FAILURE", meta=err_meta)   # save details for clients
        raise Ignore()                                 # stop task without a second write

    submission_type = submit_info.get("submit_type")
    if submission_type == "autoquiz":
        respond = autoquiz.ATQ().start(submit_info)
    elif submission_type == "assessment":
        respond = assessment.Assessment().start(submit_info)
    elif submission_type == "reassessment":
        respond = assessment.Assessment().reassessment(submit_info)
    elif submission_type == "build_canonical_topic":
        respond = canonical_topic.BuildCanonicalTopic(task=self).start(submit_info)
    elif submission_type == "generate_article":
        respond = article_generation.GenerateArticle(task=self).start(submit_info)
    else:
        err_msg = "Unknown submission type"
        # FIX: Added exc_type and exc_message for Celery backend compatibility
        err_meta = {"status": "FAILED", "message": err_msg, "exc_type": "ValueError", "exc_message": err_msg}
        logger.error("submit_job failed before dispatch: %s", err_msg)
        self.update_state(state="FAILURE", meta=err_meta)
        raise Ignore()

    # ---------- normal end-of-task handling ----------
    if respond.get("status") == "FAILED":
        # When a sub-process (autoquiz/assessment) returns FAILED,
        # propagate its message and ensure exc_type is present for Celery backend.
        respond_meta = respond.copy()
        if "exc_type" not in respond_meta:
            respond_meta["exc_type"] = "ApplicationError" # Generic error type
        if "exc_message" not in respond_meta and "message" in respond_meta:
            respond_meta["exc_message"] = respond_meta["message"]
        elif "exc_message" not in respond_meta:
            respond_meta["exc_message"] = "Task failed with unspecific error message." # Fallback

        logger.error(
            "submit_job failed: submission_type=%s doc_id=%s message=%s",
            submission_type,
            submit_info.get("doc_id"),
            respond_meta.get("message"),
        )
        self.update_state(state="FAILURE", meta=respond_meta)
        raise Ignore()

    # SUCCESS path → just return the dictionary
    # Celery will write one final SUCCESS record that contains this dict.
    logger.info(
        "submit_job succeeded: submission_type=%s doc_id=%s",
        submission_type,
        respond.get("doc_id") or submit_info.get("doc_id"),
    )
    return respond
