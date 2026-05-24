# bismillahhirahmanirahim
import os


class Config:
    def __init__(self):
        if os.getenv('OPENAI_API_KEY') is None:
            raise EnvironmentError('OPENAI_API_KEY is not set')

        env_type = os.getenv('REHAL_ENV_TYPE')
        if env_type is None:
            raise EnvironmentError('REHAL_ENV_TYPE is not set')
        if env_type not in ['PROD', 'DEV']:
            raise EnvironmentError(f'Unknown REHAL_ENV_TYPE setting : {env_type}')

        if os.getenv('MONGO_URI') is None:
            raise EnvironmentError('MONGO_URI')
        self.mongo_uri = os.getenv('MONGO_URI')

        self.eval_mongo_db_name = env_type + '_EVAL_REHAL_DB'
        self.assess_mongo_db_name = env_type + '_ASSESS_REHAL_DB'
        self.auth_mongo_db_name = env_type + '_AUTH_REHAL_DB'
        self.user_mongo_db_name = env_type + '_USER_REHAL_DB'

        self.mongo_collection_mcq_name = env_type + '_mcq'
        self.user_collection_name = 'users'
        self.openai_autoquiz_model = os.getenv('OPENAI_AUTOQUIZ_MODEL', 'gpt-4o-mini')
        self.openai_assessment_model = os.getenv('OPENAI_ASSESSMENT_MODEL', 'gpt-4o')
        self.openai_moderation_model = os.getenv('OPENAI_MODERATION_MODEL', 'omni-moderation-latest')

        self.credit_cost = {
            'quiz_generation':          5,
            'assessment_generation':    5
        }


