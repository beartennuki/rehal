#Bismillahhiramanirahim
import time
import math
from config import Config
from src.mongodbhandler import MongoDBHandler
class UpdateMCQ:
    def __init__(self):
        cfg = Config()
        self.eval_mongoIO = MongoDBHandler(cfg.eval_mongo_db_name, cfg.mongo_collection_mcq_name)
        self.asses_mongoIO = MongoDBHandler(cfg.assess_mongo_db_name, cfg.mongo_collection_mcq_name)

    def update_quiz_doc(self, update_dic):
        """
        Updates the quiz document using an optimistic concurrency control (safe-update) mechanism.
        If a version conflict is detected, the function reloads the document and tries the update again,
        up to a maximum of 100 attempts.

        update_dic must include:
          - assessment_id: The ID of the assessment.
          - eval_id: The quiz document's meta.doc_id.
          - user_id: The ID of the user.

        Returns the new version string upon success.
        """
        assessment_id = update_dic['assessment_id']
        eval_id = update_dic['eval_id']
        user_id = update_dic['user_id']

        max_attempts = 5
        attempt = 0
        new_version = None

        while attempt < max_attempts:
            attempt += 1

            # Load the current quiz document and its version.
            quiz_doc, current_version = self.eval_mongoIO.load_eval_document(eval_id)
            if not quiz_doc:
                raise Exception(f"Quiz document with doc_id {eval_id} for user {user_id} not found.")

            # Load the assessment document.
            assessment_doc, _ = self.asses_mongoIO.load_assessment_document(assessment_id)
            if not assessment_doc:
                raise Exception(f"Assessment document with assessment_id {assessment_id} not found.")

            # --- Build updated feedback ---
            feedback = quiz_doc.get("feedback", {})

            # Overall feedback: Increment finished_count.
            overall_feedback = feedback.get("overall_feedback", {})
            overall_feedback["finished_count"] = overall_feedback.get("finished_count", 0) + 1

            # Performance statistics: Append accuracy and recalc stats.
            performance = feedback.get("performance", {})
            scores = performance.get("scores", [])
            accuracy = assessment_doc.get("assessment_info", {}).get("accuracy", 0)
            scores.append(accuracy)
            performance["scores"] = scores
            performance["user_count"] = performance.get("user_count", 0) + 1
            avg_score = sum(scores) / len(scores)
            performance["avg_score"] = avg_score
            if len(scores) > 1:
                variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                performance["std_score"] = math.sqrt(variance)
            else:
                performance["std_score"] = 0

            # Question-level feedback: Update counts for correct and wrong answers.
            question_feedback = feedback.get("question_feedback", {})

            for qid in assessment_doc.get("assessment_info", {}).get("correct_qids", []):
                question_feedback[qid]["correct_count"] = question_feedback[qid].get("correct_count", 0) + 1
                question_feedback[qid]["attempt_count"] = question_feedback[qid].get("attempt_count", 0) + 1

            for qid in assessment_doc.get("assessment_info", {}).get("wrong_qids", []):
                question_feedback[qid]["incorrect_count"] = question_feedback[qid].get("incorrect_count", 0) + 1
                question_feedback[qid]["attempt_count"] = question_feedback[qid].get("attempt_count", 0) + 1

            for qid in assessment_doc.get("assessment_info", {}).get("dont_know_qids", []):
                question_feedback[qid]["unsure_count"] = question_feedback[qid].get("unsure_count", 0) + 1

            for qid in assessment_doc.get("assessment_info", {}).get("flagged_qids", []):
                question_feedback[qid]["flagged_count"] = question_feedback[qid].get("flagged_count", 0) + 1

            for qid in assessment_doc.get("assessment_info", {}).get("thumbs_up_qids", []):
                question_feedback[qid]["good_count"] = question_feedback[qid].get("good_count", 0) + 1

            # Merge the updated feedback back.
            feedback["overall_feedback"] = overall_feedback
            feedback["performance"] = performance
            feedback["question_feedback"] = question_feedback
            quiz_doc["feedback"] = feedback

            # --- Try to safely update the document ---
            try:
                new_version = self.eval_mongoIO.safe_update_document(
                    doc_id=eval_id,
                    update_fields={"feedback": feedback},
                    expected_version=current_version
                )
                # If the update succeeds, exit the loop.
                return new_version
            except Exception as e:
                # Print error and wait briefly before retrying.
                print(f"Attempt {attempt} failed: {e}")
                time.sleep(0.1)
                continue

        # If we reach here, no update succeeded after max_attempts.
        raise Exception("Failed to update quiz document after 100 attempts due to version conflicts.")

