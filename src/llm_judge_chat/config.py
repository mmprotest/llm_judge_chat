"""Configuration helpers for llm_judge_chat."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import dotenv_values, set_key
from pydantic import BaseSettings, Field, HttpUrl, validator

ENV_PATH = Path(".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    gen_base_url: HttpUrl = Field(
        "https://api.openai.com/v1", env="GEN_BASE_URL", description="Generator API base URL"
    )
    gen_api_key: str = Field(..., env="GEN_API_KEY", description="Generator API key")
    gen_model: str = Field("gpt-3.5-turbo", env="GEN_MODEL", description="Generator model name")

    judge_base_url: HttpUrl = Field(
        "https://api.openai.com/v1", env="JUDGE_BASE_URL", description="Judge API base URL"
    )
    judge_api_key: str = Field(..., env="JUDGE_API_KEY", description="Judge API key")
    judge_model: str = Field("gpt-4", env="JUDGE_MODEL", description="Judge model name")

    temperature: float = Field(0.8, env="TEMPERATURE", ge=0.0, le=2.0)
    top_p: float = Field(0.9, env="TOP_P", ge=0.0, le=1.0)
    max_tokens: int = Field(512, env="MAX_TOKENS", ge=1, le=4096)
    n_candidates: int = Field(3, env="N_CANDIDATES", ge=1, le=8)
    presence_penalty: float = Field(0.0, env="PRESENCE_PENALTY", ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, env="FREQUENCY_PENALTY", ge=-2.0, le=2.0)
    timeout_s: int = Field(60, env="TIMEOUT_S", ge=1, le=600)
    context_k: int = Field(8, env="CONTEXT_K", ge=1, le=32)
    enable_memory: bool = Field(True, env="ENABLE_MEMORY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("gen_api_key", "judge_api_key")
    def _ensure_non_empty(cls, value: str) -> str:  # noqa: D401
        """Ensure API keys are non-empty."""

        if not value or value.lower() == "changeme":
            raise ValueError("API key must be provided")
        return value


def load_settings(**overrides: Any) -> Settings:
    """Load settings with optional overrides."""

    return Settings(**overrides)


def persist_settings(updates: Dict[str, Any]) -> None:
    """Persist provided settings into the .env file."""

    if not updates:
        return

    env_path = ENV_PATH
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")

    current = dotenv_values(env_path)
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        else:
            value_str = str(value)
        if current.get(key) == value_str:
            continue
        set_key(str(env_path), key, value_str)


__all__ = ["Settings", "load_settings", "persist_settings"]
