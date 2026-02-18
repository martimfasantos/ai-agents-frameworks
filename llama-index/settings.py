import pydantic
from pathlib import Path
from pydantic_settings import BaseSettings

# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    OPENAI_API_KEY: pydantic.SecretStr 
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_EMBEDDINGS_MODEL: str = "text-embedding-ada-002"

    class Config:
        env_file = Path(__file__).parent / ".env"
        env_file_encoding = "utf-8"

settings: Settings = Settings() 
