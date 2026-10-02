import os

import pydantic
from pydantic_settings import BaseSettings

# Pydantic AI 2.43 prints a run banner on every run unless observability is
# configured. Every example imports this module, so silencing it here keeps the
# example output about the feature being demonstrated.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    OPENAI_API_KEY: pydantic.SecretStr
    OPENAI_MODEL_NAME: str = "openai-chat:gpt-4o-mini"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings: Settings = Settings()
