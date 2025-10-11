"""Tests covering configuration persistence helpers."""

from __future__ import annotations

from dotenv import dotenv_values

from llm_judge_chat import config


def test_persist_settings_normalizes_crlf_and_quotes(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_PATH", env_path)

    prompt = 'Line one with "quotes"\r\nLine two with \\slashes\\'
    config.persist_settings({"GEN_SYSTEM_PROMPT": prompt, "ENABLE_MEMORY": True})

    # File should be created and contain escaped newline sequences instead of raw CRLF.
    raw_text = env_path.read_text(encoding="utf-8")
    assert "\r" not in raw_text
    assert 'GEN_SYSTEM_PROMPT="' in raw_text
    assert "\\n" in raw_text

    values = dotenv_values(env_path)
    # python-dotenv should parse without errors and yield the escaped string.
    assert values["GEN_SYSTEM_PROMPT"] == 'Line one with "quotes"\nLine two with \\slashes\\'
    assert values["ENABLE_MEMORY"] == "true"


def test_persist_settings_is_idempotent_for_existing_values(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_PATH", env_path)

    config.persist_settings({"GEN_MODEL": "gpt-4"})
    first_content = env_path.read_text(encoding="utf-8")

    config.persist_settings({"GEN_MODEL": "gpt-4"})
    second_content = env_path.read_text(encoding="utf-8")

    assert first_content == second_content
