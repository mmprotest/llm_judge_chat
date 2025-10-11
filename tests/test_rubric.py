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
            "In-Character Fidelity": 8.0,
            "Continuity": 9.0,
            "Emotional Realism": 7.0,
            "Scene Advancement": 8.0,
            "Coherence": 6.0,
        }
    ],
)
def test_score_candidates_parsing(monkeypatch, scores) -> None:
    async def fake_chat_completion(**_: dict) -> tuple[str, dict, dict]:
        payload = {"candidates": [{"scores": scores, "rationale": "Solid."}]}
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

    weights = schemas.JudgeRubricWeights()
    expected_overall = sum(
        weights.dict()[key] * value / 10.0 for key, value in {
            "in_character_fidelity": scores["In-Character Fidelity"],
            "continuity": scores["Continuity"],
            "emotional_realism": scores["Emotional Realism"],
            "scene_advancement": scores["Scene Advancement"],
            "coherence": scores["Coherence"],
        }.items()
    )
    expected_overall = expected_overall * 10.0
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


def test_score_candidates_accepts_candidate_scores_schema(monkeypatch) -> None:
    async def fake_chat_completion(**_: dict) -> tuple[str, dict, dict]:
        payload = {
            "candidate_scores": [
                {
                    "scores": [
                        {"dimension": "In-Character Fidelity", "score": 9},
                        {"dimension": "Continuity", "score": 8},
                        {"dimension": "Emotional Realism", "score": 8},
                        {"dimension": "Scene Advancement", "score": 7},
                        {"dimension": "Coherence", "score": 9},
                    ],
                    "overall_rating": 8.4,
                    "rationale": "Strong continuation.",
                },
                {
                    "dimensions": {
                        "In-Character Fidelity": 6,
                        "Continuity": 6,
                        "Emotional Realism": 5,
                        "Scene Advancement": 5,
                        "Coherence": 7,
                    },
                    "rating": 5.8,
                    "explanation": "Generic answer.",
                },
            ]
        }
        return json.dumps(payload), {"prompt_tokens": 80, "completion_tokens": 30}, {}

    monkeypatch.setattr(judge_module, "chat_completion", fake_chat_completion)

    candidates = [
        schemas.Candidate(text="Great reply", meta={}),
        schemas.Candidate(text="Weaker reply", meta={}),
    ]
    turns = [schemas.Turn(role="user", content="Continue the scene.")]

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

    assert meta["fallback"] is False
    assert len(judged) == 2
    assert judged[0].overall == 8.4
    assert "Strong continuation" in judged[0].rationale
    assert judged[1].scores["In-Character Fidelity"] == 6.0
    assert judged[1].overall == 5.8


def test_score_candidates_partial_results_trigger_partial_fallback(monkeypatch) -> None:
    async def fake_chat_completion(**_: dict) -> tuple[str, dict, dict]:
        payload = {
            "candidates": [
                {
                    "scores": {
                        "In-Character Fidelity": 7,
                        "Continuity": 8,
                        "Emotional Realism": 7,
                        "Scene Advancement": 6,
                        "Coherence": 7,
                    },
                    "overall": 7.0,
                    "rationale": "Decent answer.",
                }
            ]
        }
        return json.dumps(payload), {"prompt_tokens": 60, "completion_tokens": 25}, {"id": "partial"}

    monkeypatch.setattr(judge_module, "chat_completion", fake_chat_completion)

    candidates = [
        schemas.Candidate(text="First", meta={}),
        schemas.Candidate(text="Second", meta={}),
    ]
    turns = [schemas.Turn(role="user", content="Tell me more."), schemas.Turn(role="assistant", content="Sure")]

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
    assert meta.get("fallback_reason") == "partial"
    assert len(judged) == 2
    assert judged[0].rationale == "Decent answer."
    assert judged[1].rationale == "Heuristic fallback"
