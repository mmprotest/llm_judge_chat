"""Thin wrapper around OpenAI-compatible chat completion APIs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - fallback when httpx missing
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

from .utils import async_retry


class ChatCompletionError(RuntimeError):
    """Raised when the chat completion request fails."""


async def _request(
    client: "httpx.AsyncClient",
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    response = await client.post("/chat/completions", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@async_retry
async def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    timeout: float = 60.0,
    seed: Optional[int] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    """Call an OpenAI-compatible chat completion endpoint."""

    if httpx is None:  # pragma: no cover - dependency missing in tests
        raise ChatCompletionError("httpx dependency is required for API calls")

    headers = {"Authorization": f"Bearer {api_key}"}
    if extra_headers:
        headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout) as client:
        data = await _request(client, payload, timeout)

    try:
        choice = data["choices"][0]
        message = choice["message"]["content"]
    except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
        raise ChatCompletionError("Malformed response from completion API") from exc

    usage = data.get("usage")
    return message, usage, data


__all__ = ["chat_completion", "ChatCompletionError"]
