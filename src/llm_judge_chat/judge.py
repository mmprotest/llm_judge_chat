"""Judge model integration."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

try:  # pragma: no cover - fallback
    import orjson
except ImportError:  # pragma: no cover
    import json as _json

    class _OrjsonFallback:
        @staticmethod
        def loads(data: str) -> Any:
            return _json.loads(data)

    orjson = _OrjsonFallback()  # type: ignore

from .openai_compat import chat_completion
from .schemas import Candidate, JudgeRubricWeights, Judged, Turn

JUDGE_PROMPT = (
    "You are an impartial dialogue judge. Score each candidate response from 0-10 "
    "for relevance, faithfulness, helpfulness, coherence, and persona/tone fit. "
    "Return a JSON object with a list under 'candidates', where each item has "
    "'scores' (per dimension), 'overall' (weighted sum), and 'rationale' "
    "(1-2 sentences, no chain-of-thought)."
)


def _build_judge_messages(
    history: Sequence[Turn],
    candidates: Sequence[Candidate],
    weights: JudgeRubricWeights,
    system_prompt: str | None,
) -> List[Dict[str, str]]:
    prompt = (system_prompt or "").strip() or JUDGE_PROMPT
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Evaluate the following dialogue and candidates.",
        },
    ]
    weight_dict = weights.as_dict()
    history_lines = ["Dialogue history:"]
    for turn in history:
        history_lines.append(f"- {turn.role}: {turn.content}")
    history_lines.append("Candidates:")
    for idx, cand in enumerate(candidates, start=1):
        history_lines.append(f"Candidate {idx}: {cand.text}")
    weight_summary = ", ".join(f"{key}: {value}" for key, value in weight_dict.items())
    history_lines.append(f"Weights: {weight_summary}")
    messages.append({"role": "user", "content": "\n".join(history_lines)})
    messages.append(
        {
            "role": "user",
            "content": (
                "Respond strictly as JSON with schema: {\"candidates\": ["
                "{\"scores\": {dim: number}, \"overall\": number, \"rationale\": string}...]}"
            ),
        }
    )
    return messages


def _fallback_scores(history: Sequence[Turn], candidates: Sequence[Candidate]) -> List[Judged]:
    """Fallback heuristic scoring when judge fails."""

    user_query = ""
    for turn in reversed(history):
        if turn.role == "user":
            user_query = turn.content
            break

    judged: List[Judged] = []
    for cand in candidates:
        overlap = len(set(user_query.lower().split()) & set(cand.text.lower().split()))
        length_score = min(10.0, max(0.0, len(cand.text) / 30.0))
        overall = (overlap + length_score) / 2
        scores = {
            "relevance": float(overlap),
            "faithfulness": float(overall),
            "helpfulness": float(overall),
            "coherence": float(length_score),
            "persona": 5.0,
        }
        judged.append(
            Judged(candidate=cand, scores=scores, overall=float(overall), rationale="Heuristic fallback")
        )
    return judged


def _compute_overall(scores: Dict[str, float], weights: JudgeRubricWeights) -> float:
    total = 0.0
    weight_dict = weights.as_dict()
    for key, weight in weight_dict.items():
        score = scores.get(key, 0.0)
        total += (score / 10.0) * weight * 10.0
    return round(total, 4)


async def score_candidates(
    *,
    base_url: str,
    api_key: str,
    model: str,
    history: Sequence[Turn],
    candidates: Sequence[Candidate],
    weights: JudgeRubricWeights,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 800,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    timeout: float,
    system_prompt: str | None = None,
) -> Tuple[List[Judged], Dict[str, Any]]:
    """Score candidates using the judge model."""

    messages = _build_judge_messages(history, candidates, weights, system_prompt)
    metadata: Dict[str, Any] = {"fallback": False}
    try:
        text, usage, raw = await chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            timeout=timeout,
        )
        payload = orjson.loads(text)
        judged: List[Judged] = []
        for cand, result in zip(candidates, payload.get("candidates", [])):
            scores = {k: float(v) for k, v in result.get("scores", {}).items()}
            if not scores:
                continue
            judged.append(
                Judged(
                    candidate=cand,
                    scores=scores,
                    overall=_compute_overall(scores, weights),
                    rationale=str(result.get("rationale", "")),
                )
            )
        if len(judged) != len(candidates):
            raise ValueError("Judge response count mismatch")
        metadata.update({"usage": usage, "raw": raw})
        return judged, metadata
    except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
        metadata.update({"fallback": True, "error": str(exc)})
        judged = _fallback_scores(history, candidates)
        return judged, metadata


__all__ = ["score_candidates"]
