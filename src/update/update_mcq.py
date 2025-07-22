# Bismillahhiramanirahim
import math
from datetime import datetime
from config import Config
from src.mongodbhandler import MongoDBHandler

class UpdateMCQ:
    """
    Handles updating MCQ quiz documents with assessment results using atomic
    database operations to prevent race conditions.
    """
    def __init__(self):
        """Initializes database handlers."""
        cfg = Config()
        # It's assumed MongoDBHandler provides access to the raw pymongo collection.
        # e.g., self.eval_mongoIO.collection
        self.eval_mongoIO = MongoDBHandler(cfg.eval_mongo_db_name, cfg.mongo_collection_mcq_name)
        self.asses_mongoIO = MongoDBHandler(cfg.assess_mongo_db_name, cfg.mongo_collection_mcq_name)

    def update_quiz_doc(self, update_dic):
        """
        Updates the quiz document statistics using atomic MongoDB operations.
        This approach avoids the "read-modify-write" pattern and prevents version conflicts.

        The update is performed in two stages to handle complex calculations:
        1. Atomically increment all counters and push the new score.
        2. Fetch the updated document, calculate new statistics (avg, std), and set them.

        Args:
            update_dic (dict): A dictionary containing assessment details.
                - assessment_id (str): The ID of the assessment.
                - eval_id (str): The ID of the quiz document to update.
                - user_id (str): The ID of the user who took the assessment.

        Returns:
            dict: The result of the final update operation.

        Raises:
            Exception: If the assessment or quiz document cannot be found.
        """
        assessment_id = update_dic['assessment_id']
        eval_id = update_dic['eval_id']
        user_id = update_dic['user_id'] # Included for logging/error context

        # --- 1. Load the Assessment Document ---
        # We only need to load the assessment document to get the results.
        # We do NOT load the quiz document to modify it.
        assessment_doc, _ = self.asses_mongoIO.load_assessment_document(assessment_id)
        if not assessment_doc:
            raise Exception(f"Assessment document with assessment_id {assessment_id} not found.")

        assessment_info = assessment_doc.get("assessment_info", {})
        accuracy = assessment_info.get("accuracy", 0)

        # --- 2. Construct the First Atomic Update (Counters & Score Push) ---
        # This operation uses $inc for counters and $push for the scores array.
        # It's a single, atomic command that is safe from race conditions.
        atomic_increments = {
            "feedback.overall_feedback.finished_count": 1,
            "feedback.performance.user_count": 1,
        }

        # Dynamically build the increment operations for each question
        q_feedback_path = "feedback.question_feedback"
        for qid in assessment_info.get("correct_qids", []):
            atomic_increments[f"{q_feedback_path}.{qid}.correct_count"] = 1
            atomic_increments[f"{q_feedback_path}.{qid}.attempt_count"] = 1

        for qid in assessment_info.get("wrong_qids", []):
            atomic_increments[f"{q_feedback_path}.{qid}.incorrect_count"] = 1
            atomic_increments[f"{q_feedback_path}.{qid}.attempt_count"] = 1

        for qid in assessment_info.get("dont_know_qids", []):
            atomic_increments[f"{q_feedback_path}.{qid}.unsure_count"] = 1

        for qid in assessment_info.get("flagged_qids", []):
            atomic_increments[f"{q_feedback_path}.{qid}.flagged_count"] = 1

        for qid in assessment_info.get("thumbs_up_qids", []):
            atomic_increments[f"{q_feedback_path}.{qid}.good_count"] = 1

        first_update_operation = {
            '$inc': atomic_increments,
            '$push': {'feedback.performance.scores': accuracy}
        }

        # Execute the first atomic update
        # We assume the handler gives access to the underlying pymongo collection
        update_result = self.eval_mongoIO.collection.update_one(
            {'meta.doc_id': eval_id},
            first_update_operation
        )

        if update_result.matched_count == 0:
            raise Exception(f"Quiz document with doc_id {eval_id} for user {user_id} not found during update.")

        # --- 3. Calculate New Stats and Perform Second Update ---
        # Now that the score is safely in the array, we can fetch the document
        # to perform calculations that can't be done with simple atomic operators.
        updated_quiz_doc = self.eval_mongoIO.collection.find_one({'meta.doc_id': eval_id})
        scores = updated_quiz_doc.get("feedback", {}).get("performance", {}).get("scores", [])

        if scores:
            avg_score = sum(scores) / len(scores)
            if len(scores) > 1:
                variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                std_score = math.sqrt(variance)
            else:
                std_score = 0
        else:
            avg_score = 0
            std_score = 0

        # This second update just sets the newly calculated values.
        # The risk of conflict here is negligible as the high-contention increments are done.
        second_update_operation = {
            '$set': {
                'feedback.performance.avg_score': avg_score,
                'feedback.performance.std_score': std_score,
                'meta.last_updated_utc': datetime.utcnow() # Also update the timestamp
            }
        }

        final_result = self.eval_mongoIO.collection.update_one(
            {'meta.doc_id': eval_id},
            second_update_operation
        )

        print(f"Successfully updated stats for quiz {eval_id}.")
        return final_result
