"""Selection utilities."""

from __future__ import annotations

from typing import Sequence

from .schemas import Judged


def select_best(judged: Sequence[Judged]) -> Judged:
    """Return the candidate with the highest overall score."""

    if not judged:
        raise ValueError("No judged candidates provided")
    return max(judged, key=lambda item: item.overall)


__all__ = ["select_best"]
