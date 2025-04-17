import uuid


class IDGenerator:
    # Mapping of genres to their 3-character suffixes
    GENRE_SUFFIXES = {
        "business": "BUS",
        "administration": "ADM",
        "finance": "FIN",
        "science": "SCI",
        "medical": "MED",
        "technology": "TEC",
        "creativity": "CRE",
        "law": "LAW",
        "culture": "CUL"
    }

    @staticmethod
    def generate_unique_id():
        """
        Generates a unique 7-character identifier using UUID4.
        """
        # Generate a random UUID4 and get the first 7 characters
        return str(uuid.uuid4().hex[:7]).upper()

    def generate_question_id(self, genre):
        """
        Generates a 10-character question ID with a 3-character genre suffix.

        Parameters:
        genre (str): The genre of the question.

        Returns:
        str: A 10-character question ID.
        """
        # Get the suffix for the given genre
        suffix = self.GENRE_SUFFIXES.get(genre.lower())
        if not suffix:
            raise ValueError(f"Genre '{genre}' is not recognized.")

        # Generate the unique part of the ID
        unique_part = self.generate_unique_id()

        # Combine the unique part with the suffix
        question_id = suffix+ unique_part
        return question_id
