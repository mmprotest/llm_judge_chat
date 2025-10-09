"""Heuristic memory management without embeddings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .schemas import Turn

NAME_PATTERN = re.compile(r"my name is ([A-Z][a-zA-Z]+)", re.IGNORECASE)
PREFERENCE_PATTERN = re.compile(r"i (?:like|prefer) ([^\.\!]+)", re.IGNORECASE)
TODO_PATTERN = re.compile(r"todo:?\s*(.*)", re.IGNORECASE)


@dataclass
class MemoryManager:
    """Maintains lightweight state about the dialogue."""

    token_budget: int = 500
    store: Dict[str, List[str]] = field(default_factory=lambda: {
        "names": [],
        "preferences": [],
        "todos": [],
    })

    def update(self, history: Iterable[Turn]) -> None:
        """Update memory store based on the most recent turn."""

        try:
            last_turn = list(history)[-1]
        except IndexError:
            return

        if last_turn.role != "user":
            return

        content = last_turn.content
        if match := NAME_PATTERN.search(content):
            name = match.group(1).strip()
            if name not in self.store["names"]:
                self.store["names"].append(name)

        if match := PREFERENCE_PATTERN.search(content):
            pref = match.group(1).strip()
            if pref and pref not in self.store["preferences"]:
                self.store["preferences"].append(pref)

        if match := TODO_PATTERN.search(content):
            todo = match.group(1).strip()
            if todo and todo not in self.store["todos"]:
                self.store["todos"].append(todo)

        self._truncate()

    def summarize(self, history: Iterable[Turn]) -> Dict[str, List[str]]:
        """Return a compact memory summary and update with latest info."""

        self.update(history)
        return {key: values[:] for key, values in self.store.items() if values}

    def _truncate(self) -> None:
        """Ensure the memory does not exceed the token budget by trimming lists."""

        total_items = sum(len(values) for values in self.store.values())
        if total_items <= self.token_budget:
            return
        for key in self.store:
            self.store[key] = self.store[key][-self.token_budget :]


__all__ = ["MemoryManager"]
