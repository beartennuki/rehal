from pathlib import Path

from dotenv import load_dotenv


def load_rehal_env():
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
    return env_path
