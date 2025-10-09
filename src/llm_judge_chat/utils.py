"""Utility functions for the application."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

try:  # pragma: no cover - provide fallback when orjson unavailable
    import orjson  # type: ignore
except ImportError:  # pragma: no cover
    class _OrjsonFallback:
        @staticmethod
        def dumps(obj: Any) -> bytes:
            return json.dumps(obj).encode("utf-8")

    orjson = _OrjsonFallback()  # type: ignore

try:  # pragma: no cover - tenacity optional for tests
    from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover
    AsyncRetrying = None  # type: ignore
    stop_after_attempt = wait_exponential = None  # type: ignore

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None  # type: ignore


def timestamp() -> str:
    """Return an ISO 8601 timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a dictionary as JSON to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        fh.write(orjson.dumps(record))
        fh.write(b"\n")


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Estimate token usage using tiktoken with graceful fallback."""

    if tiktoken is None:  # pragma: no cover - fallback path when tiktoken absent
        return max(1, len(text) // 4)

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:  # pragma: no cover - fallback
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


if AsyncRetrying is None:  # pragma: no cover - simple retry when tenacity missing

    def async_retry(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        return wrapper

else:

    def async_retry(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Retry decorator for async functions with exponential backoff."""

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async for attempt in AsyncRetrying(
                wait=wait_exponential(multiplier=1, min=1, max=8),
                stop=stop_after_attempt(3),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)
            raise RuntimeError("Unreachable")  # pragma: no cover

        return wrapper


def cancel_task(task: Optional[asyncio.Task[Any]]) -> None:
    """Cancel an asyncio task if it is running."""

    if task is None:
        return
    if not task.done():
        task.cancel()


def load_json(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""

    if not path.exists():
        return []
    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


__all__ = [
    "timestamp",
    "append_jsonl",
    "estimate_tokens",
    "async_retry",
    "cancel_task",
    "load_json",
]
