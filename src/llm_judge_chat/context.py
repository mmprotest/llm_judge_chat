"""Context preparation utilities."""

from __future__ import annotations

from typing import Iterable, List

from .memory import MemoryManager
from .schemas import DialogueState, Turn


def window_history(history: Iterable[Turn], k: int = 8) -> List[Turn]:
    """Return the last *k* turns from the dialogue history."""

    turns = list(history)
    if k <= 0:
        return turns
    return turns[-k:]


def build_context(state: DialogueState, memory_manager: MemoryManager, k: int = 8) -> dict:
    """Pack context including a rolling history window and memory summary."""

    recent = window_history(state.history, k=k)
    summary = memory_manager.summarize(state.history)
    return {
        "history": [turn.dict() for turn in recent],
        "memory_summary": summary,
    }


__all__ = ["window_history", "build_context"]
