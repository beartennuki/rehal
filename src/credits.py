import pymongo
from datetime import datetime, timezone
from config import Config


class Credit:
    """
    Manages user credits stored in MongoDB, identified solely by `user_id`.
    """
    def __init__(self):
        """
        Initializes the Credit manager using settings from Config.
        """
        self.config = Config()

        try:
            self.client = pymongo.MongoClient(
                self.config.mongo_uri,
                serverSelectionTimeoutMS=5000
            )
            # Verify connection
            self.client.admin.command('ping')

            # Access the configured database and collection
            self.db = self.client[self.config.auth_mongo_db_name]
            self.users_collection = self.db[self.config.user_collection_name]
        except pymongo.errors.ConnectionFailure as e:
            raise ConnectionError(f"Could not connect to MongoDB: {e}")
        except Exception as e:
            raise ConnectionError(f"Error during MongoDB init: {e}")

    def _get_action_cost(self, action_type: str) -> int | None:
        """Retrieves the cost of a specific action from config."""
        return self.config.credit_cost.get(action_type)

    def validate_credit(self, user_id: str, action_type: str) -> bool:
        """
        Checks if a user has enough credits for a given action and that the credits are still active.

        Args:
            user_id: The custom user identifier.
            action_type: The type of action (e.g., 'quiz_generation').

        Returns:
            True if the user has sufficient and active credits, False otherwise.
        """
        cost = self._get_action_cost(action_type)
        if cost is None:
            raise ValueError(
                f"Unknown action type '{action_type}'. Cannot determine credit cost."
            )

        now = datetime.now(timezone.utc)
        try:
            doc = self.users_collection.find_one(
                {"user_id": user_id},
                {"credits_info": 1}
            )
            if not doc or 'credits_info' not in doc:
                return False

            ci = doc['credits_info']
            remaining = ci.get('remaining', 0)
            expiry = ci.get('expired_date', now)

            # Ensure expiry is timezone-aware
            if isinstance(expiry, str):
                expiry = datetime.fromisoformat(expiry)
            if isinstance(expiry, datetime) and expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)

            # Check expiration
            if expiry < now:
                return False

            # Check expiration
            if expiry < now:
                return False

            return remaining >= cost
        except Exception as e:
            raise RuntimeError(f"Error checking credits for user {user_id}: {e}")

    def subtract_credit(self, user_id: str, action_type: str) -> tuple[bool, dict]:
        """
        Attempts to subtract credits from a user for a specific action.
        Returns:
          (True, {}) if credits were successfully subtracted,
          (False, {"reason": <explanation>}) otherwise.
        """
        # Validate action_type
        if action_type not in self.config.credit_cost:
            return False, {"reason": f"Unknown action type '{action_type}'."}

        cost = self._get_action_cost(action_type)
        if cost is None:
            return False, {"reason": f"Cannot determine credit cost for '{action_type}'."}

        now = datetime.now(timezone.utc)
        history_entry = {
            "action": action_type,
            "amount_deducted": cost,
            "timestamp": now
        }

        try:
            result = self.users_collection.update_one(
                {
                    "user_id": user_id,
                    "credits_info.remaining": {"$gte": cost},
                    "credits_info.expired_date": {"$gt": now}
                },
                {
                    "$inc": {"credits_info.remaining": -cost},
                    "$push": {"credits_info.history": history_entry}
                }
            )

            if result.matched_count == 0:
                exists = self.users_collection.count_documents({"user_id": user_id}) > 0
                reason = (
                    "Insufficient credits." if exists else "User does not exist."
                )
                return False, {"reason": reason}

            if result.modified_count == 1:
                return True, {}

            return False, {"reason": "Failed to update credits due to an unknown error."}
        except Exception as e:
            return False, {"reason": str(e)}
