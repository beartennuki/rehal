import json
import time
import copy
import random
from datetime import datetime

from openai import OpenAI
from openai import APIConnectionError, APIError, Timeout

from src.mongodbhandler import MongoDBHandler
from src.moderation import Moderation
from src.ids import IDGenerator
from config import Config


class ATQ:
    def __init__(self):
        self.cfg = Config()
        self.client = OpenAI()
        self.model = "o3-mini-2025-01-31"
        self.max_retries = 3
        self.mcq_tools = [
                            {
                                "type": "function",
                                "function": {
                                    "name": "generate_mcqs",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "genre": {
                                                "type": "string",
                                                "enum": [
                                                    "business",
                                                    "administration",
                                                    "finance",
                                                    "science",
                                                    "medical",
                                                    "technology",
                                                    "creativity",
                                                    "law",
                                                    "culture"
                                                ]
                                            },
                                            "sub_genre": {"type": "string"},
                                            "general_info": {
                                                "type": "string",
                                                "description": "Indicate what topic is covered in this quiz"
                                            },
                                            "questions": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "question": {"type": "string"},
                                                        "choices_list": {
                                                            "type": "array",
                                                            "items": {"type": "string"},
                                                            "minItems": 4,
                                                            "maxItems": 4
                                                        },
                                                        "correct_answer_index": {"type": "integer"},
                                                        "explanation": {"type": "string"},
                                                    },
                                                    "required": [
                                                        "question",
                                                        "choices",
                                                        "correct_answer",
                                                        "explanation"
                                                    ],
                                                    "additionalProperties": False
                                                },
                                                "minItems": 1
                                            }
                                        },
                                        "required": ["genre", "sub_genre", "questions"],
                                        "additionalProperties": False
                                    },
                                },
                            }
                        ]

    def give_question_packet(self, submit_info):

        topic_question = submit_info.get('topic', None)
        description_question = submit_info.get('description', None)
        num_questions = submit_info.get('num_questions', None)
        level_question = submit_info.get('level', None)

        if False in [topic_question, num_questions, level_question]:
            raise ValueError('Bad request in submit_info, missing value')

        #establishing db connection
        mongoio = MongoDBHandler(self.cfg.eval_mongo_db_name, self.cfg.mongo_collection_mcq_name)
        if mongoio.is_online() is False:
            return {"status": "FAILED", "message": "Internal MongoDB is offline"}

        text_inputs = f'{topic_question} , {description_question}'
        flagged = Moderation().text_check(text_inputs)
        if flagged:
            return {"status": "FAILED", "message": "Content flagged as inappropriate."}

        # Build system message
        messages = [
            {
                "role": "system",
                "content": (f""" 
                Generate {num_questions} different multiple-choice question about {topic_question} 
                at {level_question} level. The questions must have mix of theorytical, applied and 
                scenarios if applicable.
                In the answer list, include one correct answer, two incorrect answers, and one misleadingly 
                close but wrong answer.
                Avoid question or answer that involves mathematical equations or coding text
                """
                )
            }
        ]

        # Add additional user-provided description if available
        if description_question:
            messages.append({
                "role": "user",
                "content": description_question
            })

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                start_request_time = time.time()
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.mcq_tools
                )
                end_request_time = time.time()
                arguments = completion.choices[0].message.tool_calls[0].function.arguments
                usage = completion.usage.model_dump_json()
                try:
                    mcq_dict = json.loads(arguments)
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
                        "mcq_dict": mcq_dict,
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
        return {"status": "FAILED", "message": "Failed to generate MCQs after multiple retries."}

    def start(self, submit_info):

        respond = self.give_question_packet(submit_info)
        if respond['status'] == 'FAILED':
            return respond

        mcq_dict = respond['mcq_dict']
        request_time_info = respond['request_time_info']
        usage_dict = respond['usage_dict']
        input_dict = {
            'submit_info': submit_info,
            'model': self.model
        }

        #Creating the meta information
        meta_dict = {
            "doc_id": submit_info['doc_id'],
            "genre": mcq_dict['genre'],
            "sub_genre": mcq_dict['sub_genre'],
            "title": mcq_dict['title'],
            "type": 'mcq',
            "question_count": len(mcq_dict['questions']),
            "general_info": mcq_dict.get('general_info', ''),
            "creation_date_human": datetime.now().strftime("%B %d, %Y, %I:%M %p"),
            "creation_time": time.time(),
            "load_time_stamp": [],
            'generating_time_info': request_time_info,
            'token_usage': usage_dict,
            'input_information': input_dict
        }

        overall_feedback_dict = {
            'rating_score': 0,
            'rating_count': 0,
            'view_count': 0,
            'finished_count': 0,
            'helpful_count': 0,
            'relevant_count': 0,
            'difficulty_count': 0
        }

        genre = mcq_dict['genre']
        idgen = IDGenerator()
        qid_ls = []
        for qdict in mcq_dict['questions']:
            qid = idgen.generate_question_id(genre)
            qid_ls.append(qid)
            qdict['qid'] = qid

        question_feedback_dict = {}
        for qid in qid_ls:
            question_feedback_dict[qid] = {
                'good_count': 0,
                'unsure_count': 0,
                'flagged_count': 0,
                'correct_count': 0,
                'incorrect_count':0,
                'attempt_count': 0
            }

        performance_dick = {
            'user_count': 0,
            'scores': [],
            'avg_score': 0,
            'std_score': 0,
        }

        feedback_dict = {
            'overall_feedback': overall_feedback_dict,
            'question_feedback': question_feedback_dict,
            'performance': performance_dick
        }

        for ques in mcq_dict['questions']:
            choice_ls = ques['choices_list']
            answer_ls = [False for _ in range(len(choice_ls))]
            correct_index = ques['correct_answer_index']
            answer_ls[correct_index] = True

            combined = list(zip(choice_ls, answer_ls))
            random.shuffle(combined)
            choice_ls, answer_ls = zip(*combined)

            ques['choices_list'] = choice_ls
            ques['correct_answer_index'] = answer_ls.index(True)

        question_dict = copy.deepcopy(mcq_dict['questions'])

        mcq_doc = {
            'meta': meta_dict,
            'questions': question_dict,
            'feedback': feedback_dict,
        }

        mongoio = MongoDBHandler(self.cfg.eval_mongo_db_name, self.cfg.mongo_collection_mcq_name)
        mongoio.write_document(mcq_doc)

        return {"status": "SUCCESS",
                "message": "Autoquiz has been saved",
                "doc_id": submit_info['doc_id']}
