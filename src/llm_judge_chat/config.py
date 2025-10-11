"""Configuration helpers for llm_judge_chat."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings, Field, HttpUrl, validator

ENV_PATH = Path(".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    gen_base_url: HttpUrl = Field(
        "https://api.openai.com/v1", env="GEN_BASE_URL", description="Generator API base URL"
    )
    gen_api_key: str = Field(..., env="GEN_API_KEY", description="Generator API key")
    gen_model: str = Field("gpt-3.5-turbo", env="GEN_MODEL", description="Generator model name")
    gen_system_prompt: str = Field(
        "You are a professional assistant. Provide clear, accurate, and concise answers. Only cite sources that were explicitly mentioned by the user; never fabricate citations.",
        env="GEN_SYSTEM_PROMPT",
        description="Generator system prompt override",
    )

    judge_base_url: HttpUrl = Field(
        "https://api.openai.com/v1", env="JUDGE_BASE_URL", description="Judge API base URL"
    )
    judge_api_key: str = Field(..., env="JUDGE_API_KEY", description="Judge API key")
    judge_model: str = Field("gpt-4", env="JUDGE_MODEL", description="Judge model name")
    judge_system_prompt: str = Field(
        "You are an impartial dialogue judge for roleplay persona chatbots. Score each candidate response from 0–10 for In-Character Fidelity, Continuity, Emotional Realism, Scene Advancement, and Coherence. Return a JSON object with candidate scores, overall rating, and a concise rationale for each. Choose the best response.",
        env="JUDGE_SYSTEM_PROMPT",
        description="Judge system prompt override",
    )

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


def _serialize_env_value(value: Any) -> str:
    """Convert a Python value into a .env-friendly string."""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, str):
        value_str = value.replace("\r\n", "\n").replace("\r", "\n")
        value_str = value_str.replace("\\", "\\\\").replace("\"", "\\\"")
        value_str = value_str.replace("\n", "\\n")
        return f'"{value_str}"'

    return str(value)


def persist_settings(updates: Dict[str, Any]) -> None:
    """Persist provided settings into the .env file."""

    if not updates:
        return

    env_path = ENV_PATH
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        existing_lines = []

    serialized = {
        key: _serialize_env_value(value)
        for key, value in updates.items()
        if value is not None
    }
    if not serialized:
        return

    final_lines: List[str] = []
    handled: set[str] = set()

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            final_lines.append(line)
            continue

        key, _sep, _value = line.partition("=")
        key = key.strip()
        if key in serialized:
            final_lines.append(f"{key}={serialized[key]}")
            handled.add(key)
        else:
            final_lines.append(line)

    for key, value_str in serialized.items():
        if key not in handled:
            final_lines.append(f"{key}={value_str}")

    new_content = "\n".join(final_lines)
    if final_lines:
        new_content += "\n"

    if env_path.exists() and env_path.read_text(encoding="utf-8") == new_content:
        return

    env_path.write_text(new_content, encoding="utf-8")


__all__ = ["Settings", "load_settings", "persist_settings"]
