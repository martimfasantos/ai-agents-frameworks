import pydantic
from pydantic_settings import BaseSettings

# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    GOOGLE_API_KEY: pydantic.SecretStr
    GOOGLE_MODEL_NAME: str
    
    OPENAI_API_KEY: pydantic.SecretStr
    OPENAI_MODEL_NAME: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings: Settings = Settings() 
