# Bismillahhirahmanirahim
import json
import time
import threading
from datetime import datetime

from openai import OpenAI
from openai import APIConnectionError, APIError, Timeout

from src.mongodbhandler import MongoDBHandler
from src.update.update_mcq import UpdateMCQ
from src.credits import Credit
from config import Config


class Assessment:
    def __init__(self):
        self.cfg = Config()
        self.client = OpenAI()
        self.model = "gpt-4o"
        self.max_retries = 3
        self.assessment_tools = [
            {
                "type": "function",
                "function": {
                    "name": "assess_user_result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "weak_points": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of topics or areas where the user demonstrated weaker understanding."
                            },

                            "strong_points": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of topics or areas where the user showed strong knowledge."
                            },

                            "new_topics": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "topic": {"type": "string"},
                                        "description": {"type": "string"},
                                        "recommendation": {
                                            "type": "string",
                                            "enum": ["revision", "new"],
                                            "description": "Indicates whether this topic should be revised or if it is a new area to explore."
                                        },
                                        "level":{
                                            "type": "string",
                                            "enum": ["beginner", "intermediate", "expert"],
                                            "description": "assign an appropriate level for the user"
                                        }
                                    },

                                    "required": ["topic", "description", "recommendation"],
                                    "additionalProperties": False
                                },

                                "description": "Three new topics with descriptions and recommendations for revision or further exploration."
                            },
                            "advice": {
                                "type": "string",
                                "description": "General summary advice based on the user's results."
                            }
                        },

                        "required": ["weak_points", "strong_points", "new_topics", "advice"],
                        "additionalProperties": False
                    }
                }
            }]

    def evaluate_answer(self, eval_id, responds_dic):
        # establishing db connection
        dbname = self.cfg.eval_mongo_db_name
        clcname = self.cfg.mongo_collection_mcq_name
        mongoio = MongoDBHandler(dbname, clcname)
        if mongoio.is_online() is False:
            return {"status": "FAILED", "message": "Internal MongoDB is offline"}
        doc, doc_version = mongoio.load_eval_document(eval_id)

        correct_ls = []
        wrong_ls = []
        correct_qids = []
        wrong_qids = []
        dont_know_qids = []
        dont_know_questions = []
        thumbs_up_qids = []
        flagged_qids = []

        question_count = len(doc['questions'])

        for question_index in range(question_count):
            question_info = doc['questions'][question_index]
            correct_answer_index = question_info['correct_answer_index']
            user_answer_index = responds_dic[str(question_index)]['usr_answer']
            qid = responds_dic[str(question_index)]['qid']
            question = question_info['question']

            respond_info = {
                'question_index': question_index,
                'question': question,
                'correct_answer': question_info['choices_list'][question_info['correct_answer_index']],
                'explanation': question_info['explanation']
            }

            question_flag_info = {
                'dont_know': False,
                'flagged': False,
                'thumbs_up': False,
            }
            if responds_dic[str(question_index)]['dont_know'] is True:
                dont_know_qids.append(qid)
                dont_know_questions.append(question)
                question_flag_info['dont_know'] = True

            if responds_dic[str(question_index)]['flagged'] is True:
                flagged_qids.append(qid)
                question_flag_info['flagged'] = True

            if responds_dic[str(question_index)]['thumbs_up'] is True:
                thumbs_up_qids.append(qid)
                question_flag_info['thumbs_up'] = True

            respond_info['question_flag_info'] = question_flag_info

            if correct_answer_index == user_answer_index:
                correct_ls.append(respond_info)
                correct_qids.append(qid)

            else:
                user_answer = question_info['choices_list'][int(user_answer_index)]
                respond_info['user_answer'] = user_answer
                wrong_ls.append(respond_info)
                wrong_qids.append(qid)

        assessment_info = {
            'question_count': question_count,
            'accuracy': len(correct_qids) / question_count,
            'correct_ls': correct_ls,
            'wrong_ls': wrong_ls,
            'correct_qids': correct_qids,
            'wrong_qids': wrong_qids,
            'dont_know_qids': dont_know_qids,
            'dont_know_questions': dont_know_questions,
            'flagged_qids': flagged_qids,
            'thumbs_up_qids': thumbs_up_qids,
        }
        return assessment_info

    def generate_advice(self, user_accuracy, correct_ls, wrong_ls, dont_know_ls):

        messages = [
            {
                "role": "system",
                "content":
                    f""" 
                    Base on the following assessment resullts, provide guidance toward the user on how he can improve
                    his knowledge on the subject matter. Be straight foward in giving advice in a friendly manner. 
                    User obtained an accuracy of {user_accuracy} for the test. 
                    """
            }
        ]
        if len(correct_ls) > 0:
            correct_msg = {
                "role": "user",
                "content": f"Correct answer from the user: {correct_ls}."
            }
        else:
            correct_msg = {
                "role": "user",
                "content": f"User did not give any correct answer."
            }
        messages.append(correct_msg)

        if len(wrong_ls) > 0:
            wrong_msg = {
                "role": "user",
                "content": f"Wrong answer from the user: {wrong_ls}."
            }
            messages.append(wrong_msg)

        if len(dont_know_ls) > 0:
            dontknow_msg = {
                "role": "user",
                "content": f"The user mark the following question as dont-know"
                           f": {dont_know_ls}."
            }
            messages.append(dontknow_msg)

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                start_request_time = time.time()

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.assessment_tools
                )
                end_request_time = time.time()
                arguments = completion.choices[0].message.tool_calls[0].function.arguments
                usage = completion.usage.model_dump_json()

                try:
                    advice_dict = json.loads(arguments)
                    usage_dict = json.loads(usage)
                except json.JSONDecodeError:
                    # Retry if JSON parsing fails
                    retry_count += 1
                    time.sleep(1)
                    continue

                # Successfully parsed and validated response
                request_time_info = {
                    'start': start_request_time,
                    'end': end_request_time,
                    'durations': end_request_time - start_request_time
                }
                return {"status": "SUCCESS",
                        "advice_dict": advice_dict,
                        'usage_dict': usage_dict,
                        'request_time_info': request_time_info}

            except (APIConnectionError, APIError, Timeout) as conn_err:
                retry_count += 1
                time.sleep(1)
                continue
            except Exception as e:
                # Catch any other exception and return a failed response immediately
                return {"status": "FAILED", "message": str(e)}

        # If we've exhausted retries
        return {"status": "FAILED", "message": "Failed to generate advice after multiple retries."}

    def start(self, submit_info):

        # Collect all necessary info so can fail fast
        user_id = submit_info.get('user_id', None)
        eval_id = submit_info.get('eval_id', None)
        assessment_id = submit_info.get('assessment_id', None)
        responds_dic = submit_info.get('responds', None)

        credits_obj = Credit()
        credit_validation = credits_obj.validate_credit(user_id, 'assessment_generation')
        if not credit_validation:
            return {"status": "FAILED", "message": "Not enough credit or expired"}

        assessment_info = self.evaluate_answer(eval_id, responds_dic)

        acc = assessment_info['accuracy']
        correct_ls = assessment_info['correct_ls']
        wrong_ls = assessment_info['wrong_ls']
        dont_know_ls = assessment_info['dont_know_questions']

        if responds_dic["summaryType"] == 'personalized':
            respond = self.generate_advice(acc, correct_ls, wrong_ls, dont_know_ls)
            if respond['status'] == 'FAILED':
                return respond
            respond['advice_dict']['requested'] = True

        elif responds_dic["summaryType"] == 'quick':
            respond = {
                'advice_dict':          {'requested': False},
                'request_time_info':    {'requested': False},
                'usage_dict':           {'requested': False},
            }
        else:
            raise ValueError(f'Unknown summaryType: {responds_dic["summaryType"]}')

        advice_dict = respond['advice_dict']
        request_time_info = respond['request_time_info']
        usage_dict = respond['usage_dict']

        meta_dict = {
            "eval_id": eval_id,
            'assessment_id': assessment_id,
            "creation_date_human": datetime.now().strftime("%B %d, %Y, %I:%M %p"),
            "creation_time": time.time(),
            'generating_time_info': request_time_info,
            'token_usage': usage_dict,
            'load_time_stamp': [],
            'user_id': user_id
        }

        # Writing assessment to mongodb
        assessment_doc = {
            'meta': meta_dict,
            'assessment_info': assessment_info,
            'advice_dict': advice_dict,
        }

        mongoio = MongoDBHandler(self.cfg.assess_mongo_db_name, self.cfg.mongo_collection_mcq_name)
        mongoio.write_document(assessment_doc)
        result_dic = {
            "status": "SUCCESS",
            "message": "Assessment dict has been saved",
            "assessment_id": assessment_id,
            'eval_id': eval_id,
            'user_id': user_id}

        #TODO make this a celery task
        update_obj = UpdateMCQ()
        thread = threading.Thread(target=update_obj.update_quiz_doc, args=(result_dic,))
        thread.daemon = True
        thread.start()

        _ = Credit().subtract_credit(user_id, 'assessment_generation')

        return result_dic

    def reassessment(self, resubmit_info):

        assessment_id = resubmit_info.get('assessment_id', None)
        user_id = resubmit_info.get('user_id', None)

        credits_obj = Credit()
        credit_validation = credits_obj.validate_credit(user_id, 'assessment_generation')
        if not credit_validation:
            return {"status": "FAILED", "message": "Not enough credit or expired"}

        dbname = self.cfg.assess_mongo_db_name
        clcname = self.cfg.mongo_collection_mcq_name
        mongoio = MongoDBHandler(dbname, clcname)
        assessment_doc, version = mongoio.load_assessment_document(assessment_id)

        eval_id = assessment_doc['meta']['eval_id']
        creation_date_human = assessment_doc['meta']['creation_date_human']
        creation_time = assessment_doc['meta']['creation_time']

        assessmend_info = assessment_doc['assessment_info']
        acc = assessmend_info['accuracy']
        correct_ls = assessmend_info['correct_ls']
        wrong_ls = assessmend_info['wrong_ls']
        dont_know_ls = assessmend_info['dont_know_questions']

        respond = self.generate_advice(acc, correct_ls, wrong_ls, dont_know_ls)

        if respond['status'] == 'FAILED':
            return respond
        respond['advice_dict']['requested'] = True

        advice_dict = respond['advice_dict']
        request_time_info = respond['request_time_info']
        usage_dict = respond['usage_dict']

        meta_dict = {
            "eval_id": eval_id,
            'assessment_id': assessment_id,
            "creation_date_human": creation_date_human,
            "creation_time": creation_time,
            'generating_time_info': request_time_info,
            'token_usage': usage_dict,
            'load_time_stamp': [time.time()],
            'user_id': user_id
        }

        #TODO make it try 3 times if update failed
        # updating meta
        new_ctr_version = mongoio.safe_update_document(
            doc_id=assessment_id,
            update_fields={"meta": meta_dict},
            expected_version=version
        )

        # updating advice info
        _ = mongoio.safe_update_document(
            doc_id=assessment_id,
            update_fields={"advice_dict": advice_dict},
            expected_version=new_ctr_version
        )

        result_dic = {
            "status": "SUCCESS",
            "message": "Advice dict has been saved",
            "assessment_id": assessment_id,
            'eval_id': eval_id,
            'user_id': user_id}

        _ = Credit().subtract_credit(user_id, 'assessment_generation')

        return result_dic

