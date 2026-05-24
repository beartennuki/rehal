import time
import copy
import random
from datetime import datetime

from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, RateLimitError
from pydantic import BaseModel

from src.mongodbhandler import MongoDBHandler
from src.moderation import Moderation
from src.ids import IDGenerator
from src.credits import Credit
from config import Config


class MCQQuestion(BaseModel):
    question: str
    choices_list: list[str]
    correct_answer_index: int
    explanation: str


class MCQResponse(BaseModel):
    title: str
    genre: str
    sub_genre: str
    general_info: str = ""
    questions: list[MCQQuestion]


class ATQ:
    def __init__(self):
        self.cfg = Config()
        self.client = OpenAI()
        self.model = self.cfg.openai_autoquiz_model
        self.max_retries = 3
        self.valid_genres = {
            "business",
            "administration",
            "finance",
            "science",
            "medical",
            "technology",
            "creativity",
            "law",
            "culture",
        }

    def _generate_structured_response(self, instructions, user_context):
        response = self.client.responses.parse(
            model=self.model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": user_context,
                }
            ],
            text_format=MCQResponse,
        )

        refusal = self._extract_refusal(response)
        if refusal:
            raise ValueError(f"Model refused request: {refusal}")

        parsed = getattr(response, "output_parsed", None)
        if not parsed:
            raise ValueError("Model returned no structured output.")

        self._validate_mcq_response(parsed)

        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if usage else {}
        return parsed.model_dump(), usage_dict

    @staticmethod
    def _format_openai_error(exc):
        body = getattr(exc, "body", None)
        request_id = getattr(exc, "request_id", None)
        detail = body if body is not None else str(exc)

        if request_id:
            return f"OpenAI request failed: {detail} (request_id={request_id})"
        return f"OpenAI request failed: {detail}"

    @staticmethod
    def _extract_refusal(response):
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "message":
                continue

            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "refusal":
                    return content.refusal

        return None

    def _validate_mcq_response(self, parsed):
        normalized_genre = parsed.genre.strip().lower()
        if normalized_genre not in self.valid_genres:
            raise ValueError(f"Model returned unsupported genre: {parsed.genre}")
        parsed.genre = normalized_genre

        parsed.sub_genre = parsed.sub_genre.strip()
        parsed.title = parsed.title.strip()
        parsed.general_info = parsed.general_info.strip()

        if not parsed.questions:
            raise ValueError("Model returned no questions.")

        for question in parsed.questions:
            question.question = question.question.strip()
            question.explanation = question.explanation.strip()
            question.choices_list = [choice.strip() for choice in question.choices_list]

            if len(question.choices_list) != 4:
                raise ValueError("Each question must contain exactly 4 choices.")

            if question.correct_answer_index < 0 or question.correct_answer_index >= len(question.choices_list):
                raise ValueError("Question returned an invalid correct_answer_index.")

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
        try:
            flagged = Moderation().text_check(text_inputs)
        except RuntimeError as exc:
            return {"status": "FAILED", "message": str(exc)}

        if flagged is True:
            return {"status": "FAILED", "message": "Content flagged as inappropriate."}

        instructions = (
            f"Generate {num_questions} different multiple-choice questions about {topic_question} "
            f"at {level_question} level. Mix theoretical, applied, and scenario-based questions when applicable. "
            "Each question must have exactly four choices in choices_list, with exactly one correct answer index. "
            "Include one correct answer, two clearly incorrect answers, and one plausible but wrong distractor. "
            "Avoid questions or answers involving mathematical equations or coding text. "
            "Return valid JSON matching the provided schema."
        )
        user_context = description_question or f"Topic: {topic_question}"

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                start_request_time = time.time()
                mcq_dict, usage_dict = self._generate_structured_response(
                    instructions=instructions,
                    user_context=user_context,
                )
                end_request_time = time.time()

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

            except BadRequestError as exc:
                return {"status": "FAILED", "message": self._format_openai_error(exc)}
            except (APIConnectionError, APITimeoutError, RateLimitError):
                retry_count += 1
                time.sleep(1)
                continue
            except APIError as exc:
                return {"status": "FAILED", "message": self._format_openai_error(exc)}
            except Exception as e:
                # Catch any other exception and return a failed response immediately
                return {"status": "FAILED", "message": str(e)}

        # If we've exhausted retries
        return {"status": "FAILED", "message": "Failed to generate MCQs after multiple retries."}

    def start(self, submit_info):

        user_id = submit_info['user_id']
        credits_obj = Credit()
        credit_validation = credits_obj.validate_credit(user_id, 'quiz_generation')
        if not credit_validation:
            return {"status": "FAILED", "message": "Not enough credit or expired"}

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

        _ = Credit().subtract_credit(user_id, 'quiz_generation')

        return {"status": "SUCCESS",
                "message": "Autoquiz has been saved",
                "doc_id": submit_info['doc_id']}
