from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", extra="ignore", case_sensitive=False
    )

    app_name: str = "HelpDeskAI"
    environment: str = "development"
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'helpdesk.db'}"
    admin_api_key: str = ""

    # Used from Stage 3 onward
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()