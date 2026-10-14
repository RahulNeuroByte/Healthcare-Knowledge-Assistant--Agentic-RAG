from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    serpapi_key: str = Field(
        default="", validation_alias=AliasChoices("SERPAPIKEY", "SERPAPI_KEY")
    )
    data_path: str = str(ROOT / "data" / "healthcare_faqs.json")
    index_path: str = str(ROOT / "data" / "faiss_index")
    feedback_path: str = str(ROOT / "data" / "feedback.jsonl")
    top_k: int = 4
    max_retries: int = 1
    api_url: str = "http://localhost:8000"


settings = Settings()
