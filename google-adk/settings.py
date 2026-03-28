from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    GOOGLE_API_KEY: SecretStr
    GOOGLE_MODEL_NAME: str = "gemini-2.5-flash"
    OPENAI_API_KEY: SecretStr = SecretStr("")  # used by 09_litellm.py
    OPENAI_MODEL_NAME: str = "openai/gpt-4o-mini"  # LiteLLM format: "<provider>/<model>"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings: Settings = Settings()
