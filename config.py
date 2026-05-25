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
        self.canonical_topic_mongo_db_name = os.getenv(
            'CANONICAL_TOPIC_MONGO_DB_NAME',
            os.getenv('POOL_MONGO_DB_NAME', env_type + '_CANONICAL_TOPIC_REHAL_DB')
        )
        self.canonical_topic_collection_name = os.getenv(
            'CANONICAL_TOPIC_COLLECTION_NAME',
            os.getenv('POOL_COLLECTION_NAME', env_type + '_canonical_topic')
        )
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')

        self.openai_autoquiz_model = os.getenv('OPENAI_AUTOQUIZ_MODEL', 'gpt-4o-mini')
        self.openai_assessment_model = os.getenv('OPENAI_ASSESSMENT_MODEL', 'gpt-4o')
        self.openai_moderation_model = os.getenv('OPENAI_MODERATION_MODEL', 'omni-moderation-latest')
        self.openai_canonical_topic_json_model = os.getenv(
            'OPENAI_CANONICAL_TOPIC_JSON_MODEL',
            os.getenv('OPENAI_POOL_JSON_MODEL', 'gpt-4.1-mini')
        )
        self.openai_canonical_topic_writer_model = os.getenv(
            'OPENAI_CANONICAL_TOPIC_WRITER_MODEL',
            os.getenv('OPENAI_POOL_WRITER_MODEL', 'gpt-4.1')
        )
        self.openai_canonical_topic_embedding_model = os.getenv(
            'OPENAI_CANONICAL_TOPIC_EMBEDDING_MODEL',
            os.getenv('OPENAI_POOL_EMBEDDING_MODEL', 'text-embedding-3-small')
        )

        self.canonical_topic_min_subtopics = int(os.getenv('CANONICAL_TOPIC_MIN_SUBTOPICS', os.getenv('POOL_MIN_SUBTOPICS', '4')))
        self.canonical_topic_max_subtopics = int(os.getenv('CANONICAL_TOPIC_MAX_SUBTOPICS', os.getenv('POOL_MAX_SUBTOPICS', '8')))
        self.canonical_topic_min_sources_per_subtopic = int(
            os.getenv('CANONICAL_TOPIC_MIN_SOURCES_PER_SUBTOPIC', os.getenv('POOL_MIN_SOURCES_PER_SUBTOPIC', '3'))
        )
        self.canonical_topic_max_results_per_search = int(
            os.getenv('CANONICAL_TOPIC_MAX_RESULTS_PER_SEARCH', os.getenv('POOL_MAX_RESULTS_PER_SEARCH', '10'))
        )
        self.canonical_topic_search_max_tier = int(
            os.getenv('CANONICAL_TOPIC_SEARCH_MAX_TIER', os.getenv('POOL_SEARCH_MAX_TIER', '3'))
        )
        self.canonical_topic_raw_content_limit = int(
            os.getenv('CANONICAL_TOPIC_RAW_CONTENT_LIMIT', os.getenv('POOL_RAW_CONTENT_LIMIT', '12000'))
        )

        self.credit_cost = {
            'quiz_generation':          5,
            'assessment_generation':    5
        }
