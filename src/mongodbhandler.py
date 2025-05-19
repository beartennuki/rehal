from pymongo import MongoClient, errors
from bson import ObjectId
from uuid import uuid4
import time

from config import Config

class MongoDBHandler:
    def __init__(self, db_name, collection_name):
        """
        Initializes the MongoDB client, and selects the specified database and collection.

        :param uri: MongoDB connection URI.
        :param db_name: Name of the database.
        :param collection_name: Name of the collection.
        """
        # Set a timeout (in milliseconds) for the initial connection attempt
        cfg = Config()
        self.client = MongoClient(cfg.mongo_uri, serverSelectionTimeoutMS=5000)
        self.db_name = db_name
        self.db = self.client[self.db_name]
        self.collection = self.db[collection_name]

    def write_document(self, document):
        """
        Inserts a document into the collection.

        :param document: Dictionary representing the document to insert.
        :return: The inserted document's id.
        """
        document['control'] = {
            'version': uuid4().hex,
            'time_ls': [time.time()]
        }
        result = self.collection.insert_one(document)
        return result.inserted_id

    def load_eval_document(self, doc_id, section=None):
        """
        Loads (retrieves) a document by its `doc_id` stored in `meta.doc_id`.
        Can return the whole document or just a specific section along with its version.

        :param doc_id: The document ID stored inside `meta.doc_id`
        :param section: The section to retrieve (e.g., "feedback", "input", etc.).
                        If None, retrieves the full document.
        :return: A tuple (data, version) where:
                 - `data` is the requested section (or full document).
                 - `version` is the document's version.
                 - Returns (None, None) if the document is not found.
        """
        # Construct the query
        query = {"meta.doc_id": doc_id}
        if section is None:
            document = self.collection.find_one(query)
            if not document:
                return None, None

            version = document['control']['version']
            return document, version
        else:
            projection = {"control": 1, section: 1, "_id": 0}
            document = self.collection.find_one(query, projection)
            if not document:
                return None, None
            version = document['control']['version']
            section_document = document[section]
            return section_document, version

    def load_assessment_document(self, assessment_id, section=None):
        """
        Loads (retrieves) a document by its `doc_id` stored in `meta.doc_id`.
        Can return the whole document or just a specific section along with its version.

        :param doc_id: The document ID stored inside `meta.doc_id`
        :param section: The section to retrieve (e.g., "feedback", "input", etc.).
                        If None, retrieves the full document.
        :return: A tuple (data, version) where:
                 - `data` is the requested section (or full document).
                 - `version` is the document's version.
                 - Returns (None, None) if the document is not found.
        """
        # Construct the query
        query = {"meta.assessment_id": assessment_id}
        if section is None:
            document = self.collection.find_one(query)
            if not document:
                return None, None

            version = document['control']['version']
            return document, version
        else:
            projection = {"control": 1, section: 1, "_id": 0}
            document = self.collection.find_one(query, projection)
            if not document:
                return None, None
            version = document['control']['version']
            section_document = document[section]
            return section_document, version

    def document_exists(self, doc_id):
        """
        Checks whether a document exists for the given id.

        :param doc_id: The id of the document (as a string or ObjectId).
        :return: True if the document exists, otherwise False.
        """
        try:
            object_id = ObjectId(doc_id)
        except Exception:
            object_id = doc_id
        return self.collection.find_one({"_id": object_id}) is not None

    def is_online(self):
        """
        Checks if the MongoDB server is online and the specified database is accessible.

        :return: True if MongoDB is online and the database is accessible, otherwise False.
        """
        try:
            # Ping the server
            self.client.admin.command('ping')

            # Optional: check if the database appears in the list of database names.
            # Note: A new database with no collections may not appear, so this check is optional.
            if self.db_name not in self.client.list_database_names():
                # You might choose to return False here if the database must pre-exist.
                pass

            return True
        except errors.PyMongoError as e:
            print("Error connecting to MongoDB:", e)
            return False

    def safe_update_document(self, doc_id, update_fields, expected_version):
        """
        Safely updates a document using optimistic concurrency control (eager version control).

        It updates the document identified by either meta.assessment_id or meta.eval_id
        only if its current version (stored in control.version) matches the provided expected_version.
        If successful, it merges the update_fields, generates a new version, and appends the current time to control.time_ls.

        :param doc_id: The document ID (either assessment_id or eval_id).
        :param update_fields: A dict containing the fields to update.
        :param expected_version: The version of the document that the caller expects.
        :return: The new version string if the update is successful.
        :raises Exception: If no document is updated (i.e., due to a version conflict or document not found).
        """
        new_version = uuid4().hex

        # Try matching eval_id first, if not, fall back to assessment_id
        filter_query = {
            "$or": [
                {"meta.eval_id": doc_id},
                {"meta.assessment_id": doc_id}
            ],
            "control.version": expected_version
        }

        update_set = update_fields.copy()
        update_set["control.version"] = new_version

        update_op = {
            "$set": update_set,
            "$push": {"control.time_ls": time.time()}
        }

        result = self.collection.update_one(filter_query, update_op)

        if result.modified_count == 0:
            raise Exception("Safe update failed due to version conflict or document not found.")

        return new_version

