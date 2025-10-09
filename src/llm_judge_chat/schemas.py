"""Pydantic data models used throughout the application."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:  # pragma: no cover - fallback for environments without pydantic
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    class _FieldSpec:
        def __init__(self, default: Any = None, default_factory: Any = None):
            self.default = default
            self.default_factory = default_factory

    def Field(default: Any = None, default_factory: Any | None = None, **_: Any) -> Any:
        return _FieldSpec(default, default_factory)

    class BaseModel:
        def __init__(self, **data: Any) -> None:
            for name, annotation in self.__annotations__.items():
                value = data.get(name, getattr(self.__class__, name, None))
                if isinstance(value, _FieldSpec):
                    spec = value
                    if spec.default_factory is not None:
                        value = spec.default_factory()
                    else:
                        value = spec.default
                if isinstance(value, _FieldSpec):
                    value = value.default
                setattr(self, name, value)

        def dict(self) -> Dict[str, Any]:
            return {name: getattr(self, name) for name in self.__annotations__}

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            return f"{self.__class__.__name__}({self.dict()!r})"


class Turn(BaseModel):
    """Represents a single message turn in the dialogue."""

    role: Literal["user", "assistant", "tool"]
    content: str
    meta: Optional[Dict[str, Any]] = None


class DialogueState(BaseModel):
    """Captures the full dialogue state passed between components."""

    history: List[Turn] = Field(default_factory=list)
    memory: Dict[str, Any] = Field(default_factory=dict)
    scratch: Optional[Dict[str, Any]] = None


class Candidate(BaseModel):
    """A generated candidate response with metadata."""

    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class Judged(BaseModel):
    """A candidate paired with judge scores and rationale."""

    candidate: Candidate
    scores: Dict[str, float]
    overall: float
    rationale: str


class JudgeRubricWeights(BaseModel):
    """Default weighting for rubric dimensions."""

    relevance: float = 0.25
    faithfulness: float = 0.25
    helpfulness: float = 0.25
    coherence: float = 0.20
    persona: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        """Return the weights as a plain dictionary."""

        return self.dict()


__all__ = [
    "Turn",
    "DialogueState",
    "Candidate",
    "Judged",
    "JudgeRubricWeights",
]
