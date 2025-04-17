# bismillahhirahmanirahim
import os

class Config:
    def __init__(self):
        if os.getenv('OPENAI_API_KEY') is None:
            raise EnvironmentError('OPENAI_API_KEY is not set')

        env_type = os.getenv('REHAL_ENV_TYPE')
        if env_type is None:
            raise EnvironmentError('REHAL_ENV_TYPE is not set')
        if env_type not in ['PROD', 'TEST']:
            raise EnvironmentError(f'Unknown REHAL_ENV_TYPE setting : {env_type}')

        self.eval_mongo_db_name = env_type + '_EVAL_REHAL_DB'
        self.assess_mongo_db_name = env_type + '_ASSESS_REHAL_DB'

        self.mongo_collection_mcq_name = env_type + '_mcq'

        self.mongo_uri = "mongodb://localhost:27017"

