"""Application configuration loaded from environment variables.

Centralises every tunable parameter so that no module hard-codes paths,
URLs, or hyper-parameters. Values are validated at startup.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Enterprise AI Document Intelligence")
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_base_url: str = Field(default="http://localhost:8000")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_llm_model: str = Field(default="llama3.2")
    ollama_embed_model: str = Field(default="nomic-embed-text")
    ollama_timeout: int = Field(default=120)

    data_dir: Path = Field(default=Path("./data"))
    upload_dir: Path = Field(default=Path("./data/uploads"))
    chroma_dir: Path = Field(default=Path("./data/chroma"))
    sqlite_path: Path = Field(default=Path("./data/metadata.db"))

    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    dense_weight: float = Field(default=0.7)
    sparse_weight: float = Field(default=0.3)
    retrieval_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=5)

    agent_max_iterations: int = Field(default=3)

    chroma_collection: str = Field(default="documents")

    def ensure_directories(self) -> None:
        """Create filesystem directories that the app requires."""
        for path in (self.data_dir, self.upload_dir, self.chroma_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton :class:`Settings` instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
