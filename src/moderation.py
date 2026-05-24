import random
import time

from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError


class Moderation:
    def __init__(self):
        from config import Config

        cfg = Config()
        self.client = OpenAI()
        self.model = cfg.openai_moderation_model
        self.max_retries = 5
        self.base_delay_seconds = 1.0

    def text_check(self, text):
        retryable_errors = (APIConnectionError, APIError, APITimeoutError, RateLimitError)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.moderations.create(
                    model=self.model,
                    input=text
                )
                return response.results[0].flagged
            except retryable_errors as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        "Moderation service is temporarily unavailable. Please retry shortly."
                    ) from exc

                delay = self.base_delay_seconds * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.5)
                time.sleep(delay + jitter)
