from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # NoDecode stops pydantic-settings from JSON-decoding this value before
    # validation runs — without it, a plain comma-separated .env string
    # (not a JSON array) raises a SettingsError before our validator below
    # ever sees it.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    use_stubs: bool = True
    model_artifact_path: str = "./artifacts/current"

    llm_provider: Literal["gemini", "none"] = "gemini"
    gemini_api_key: str | None = None

    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, v: object) -> object:
        """Accept a plain comma-separated string from .env, e.g.
        CORS_ORIGINS=http://localhost:5173,http://localhost:3000"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()