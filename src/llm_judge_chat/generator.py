"""Candidate generation logic."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, Iterable, List, Sequence

from .openai_compat import chat_completion
from .schemas import Candidate, Turn

STYLE_ADAPTERS: Dict[str, str] = {
    "pragmatic": "Adopt a pragmatic, solution-focused tone.",
    "coaching": "Use an encouraging coaching voice, focusing on next steps.",
    "skeptical": "Gently question assumptions and verify facts before advising.",
    "concise": "Be succinct without losing clarity or key details.",
}

SYSTEM_PROMPT = (
    "You are a professional assistant. Provide clear, accurate, and concise answers. "
    "Only cite sources that were explicitly mentioned by the user; never fabricate citations."
)


async def _single_candidate(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    decoding: Dict[str, Any],
) -> Candidate:
    text, usage, raw = await chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=decoding["temperature"],
        top_p=decoding["top_p"],
        max_tokens=decoding.get("max_tokens"),
        presence_penalty=decoding.get("presence_penalty", 0.0),
        frequency_penalty=decoding.get("frequency_penalty", 0.0),
        timeout=decoding.get("timeout", 60.0),
        seed=decoding.get("seed"),
    )
    meta = {
        "decoding": decoding,
        "usage": usage,
        "raw": raw,
    }
    return Candidate(text=text.strip(), meta=meta)


def _build_messages(history: Sequence[Turn], style_instruction: str, memory_summary: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": f"{SYSTEM_PROMPT} {style_instruction}"}]
    if memory_summary:
        summary_lines = [
            "Known user facts:" ,
        ] + [f"- {key}: {', '.join(value)}" for key, value in memory_summary.items()]
        messages.append({"role": "system", "content": "\n".join(summary_lines)})
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    return messages


async def generate_candidates(
    *,
    base_url: str,
    api_key: str,
    model: str,
    history: Sequence[Turn],
    memory_summary: Dict[str, Any],
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    presence_penalty: float,
    frequency_penalty: float,
    timeout: float,
) -> List[Candidate]:
    """Generate *n* diverse candidate replies."""

    styles = list(STYLE_ADAPTERS.keys())
    tasks = []
    for idx in range(n):
        style = styles[idx % len(styles)]
        style_instruction = STYLE_ADAPTERS[style]
        messages = _build_messages(history, style_instruction, memory_summary)
        decoding = {
            "temperature": min(2.0, max(0.0, temperature + random.uniform(-0.1, 0.1))),
            "top_p": min(1.0, max(0.1, top_p + random.uniform(-0.05, 0.05))),
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "timeout": timeout,
            "seed": random.randint(1, 10_000_000),
            "style": style,
        }
        tasks.append(
            _single_candidate(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                decoding=decoding,
            )
        )

    return await asyncio.gather(*tasks)


__all__ = ["generate_candidates", "STYLE_ADAPTERS"]
