import logging
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # ---- LLM ----
    openai_api_base: str
    openai_api_key: str = "EMPTY"
    openai_model: str
    openai_max_tokens: int = 8192

    # ---- Milvus ----
    milvus_uri: str
    milvus_collection_name: str = "rag_collection"

    # ---- Embedding ----
    embedding_api_url: str
    embedding_model_name: str
    embedding_api_key: str



    # ---- Reranker ----
    reranker_api_base: str
    reranker_model_name: str = "bge-m3-reranker"
    reranker_api_key: str = "EMPTY"

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"

    # ---- MongoDB ----
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "db"
    mongodb_collection_name: str = "chunks"

    # ---- MCP Web Search ----
    mcp_server_url: str = "https://mcp.api-inference.modelscope.net/0acbd0efb9594a/mcp"

    # ---- LangSmith ----
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "adaptive-rag"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
