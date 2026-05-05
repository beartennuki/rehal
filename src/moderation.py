from openai import OpenAI
class Moderation:
    def __init__(self):
        self.client = OpenAI()
        self.model = "omni-moderation-latest"

    def text_check(self, text):

        response = self.client.moderations.create(
            model=self.model,
            input=text
        )
        return response.results[0].flagged