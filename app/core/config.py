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

    # --- RAG (Stage 2) ---
    kb_dir: Path = BASE_DIR / "data" / "knowledge_base"
    index_dir: Path = BASE_DIR / "data" / "index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800      # characters
    chunk_overlap: int = 120   # characters
    retrieval_top_k: int = 4
    retriever_name: str = "dense_v1"

    # --- LLM (Stage 3) ---
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()