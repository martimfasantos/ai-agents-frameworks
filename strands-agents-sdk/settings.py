"""Settings module for Strands Agents SDK examples.

Uses pydantic-settings to load configuration from environment variables
or a .env file. Strands defaults to Amazon Bedrock as the model provider,
but you can configure OpenAI or Anthropic as alternatives.
"""

from typing import Optional

import pydantic
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AWS credentials (for Amazon Bedrock - the default provider)
    AWS_ACCESS_KEY_ID: Optional[pydantic.SecretStr] = None
    AWS_SECRET_ACCESS_KEY: Optional[pydantic.SecretStr] = None
    AWS_DEFAULT_REGION: str = "us-west-2"

    # OpenAI (alternative provider)
    OPENAI_API_KEY: Optional[pydantic.SecretStr] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"

    # Anthropic (alternative provider)
    ANTHROPIC_API_KEY: Optional[pydantic.SecretStr] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings: Settings = Settings()
