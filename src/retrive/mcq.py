#bismillahhirahmanirahimm
from src.mongodbhandler import MongoDBHandler
from config import Config
class MCQDoc:
    def __init__(self):
        self.cfg = Config()
        self.dbcollection = 'mcq'

    def load_mcq(self, doc_info):
        doc_id = doc_info.get('doc_id', None)
        if doc_id is None:
            return {"status": "FAILURE", "message": 'doc_id is missing'}

            # establishing db connection
        mongoio = MongoDBHandler(self.cfg.eval_mongo_db_name, self.cfg.mongo_collection_mcq_name)

        if mongoio.is_online() is False:
            return {"status": "FAILURE", "message": "Internal MongoDB is offline"}

        section = doc_info.get('section', None)
        doc, _ = mongoio.load_eval_document(doc_id, section=section)
        if doc is None:
            return {"status": "FAILURE",
                    "message": f'No document with doc_id:{doc_id} or section:{section}',
                    "doc": None}
        else:
            return {"status": "SUCCESS",
                    "doc": doc}