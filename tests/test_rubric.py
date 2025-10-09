from __future__ import annotations

import asyncio

import json
import pytest

from llm_judge_chat import schemas
from llm_judge_chat.judge import score_candidates
import llm_judge_chat.judge as judge_module


@pytest.mark.parametrize(
    "scores",
    [
        {
            "relevance": 8.0,
            "faithfulness": 9.0,
            "helpfulness": 7.0,
            "coherence": 8.0,
            "persona": 6.0,
        }
    ],
)
def test_score_candidates_parsing(monkeypatch, scores) -> None:
    async def fake_chat_completion(**_: dict) -> tuple[str, dict, dict]:
        payload = {"candidates": [{"scores": scores, "overall": 1.0, "rationale": "Solid."}]}
        return json.dumps(payload), {"prompt_tokens": 100, "completion_tokens": 20}, {}

    monkeypatch.setattr(judge_module, "chat_completion", fake_chat_completion)

    candidates = [schemas.Candidate(text="Candidate answer", meta={})]
    turns = [schemas.Turn(role="user", content="What is the plan?")]
    judged, meta = asyncio.run(
        score_candidates(
            base_url="http://localhost",
            api_key="test",
            model="judge",
            history=turns,
            candidates=candidates,
            weights=schemas.JudgeRubricWeights(),
            timeout=5.0,
        )
    )

    expected_overall = sum(
        schemas.JudgeRubricWeights().dict()[key] * value for key, value in scores.items()
    )
    assert pytest.approx(judged[0].overall, rel=1e-3) == expected_overall
    assert meta["fallback"] is False
    assert judged[0].rationale == "Solid."


def test_score_candidates_fallback(monkeypatch) -> None:
    async def failing_chat_completion(**_: dict) -> tuple[str, dict, dict]:
        raise RuntimeError("network error")

    monkeypatch.setattr(judge_module, "chat_completion", failing_chat_completion)

    candidates = [schemas.Candidate(text="fallback", meta={}), schemas.Candidate(text="another", meta={})]
    turns = [schemas.Turn(role="user", content="Explain."), schemas.Turn(role="assistant", content="Sure")]

    judged, meta = asyncio.run(
        score_candidates(
            base_url="http://localhost",
            api_key="test",
            model="judge",
            history=turns,
            candidates=candidates,
            weights=schemas.JudgeRubricWeights(),
            timeout=5.0,
        )
    )

    assert meta["fallback"] is True
    assert len(judged) == 2
    assert all(item.rationale == "Heuristic fallback" for item in judged)
