"""Judge model integration."""

from __future__ import annotations

import json
import re
import math
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    "You are an impartial dialogue judge for roleplay persona chatbots. "
    "Score each candidate response from 0–10 for: In-Character Fidelity, Continuity, "
    "Emotional Realism, Scene Advancement, and Coherence. Return a JSON object "
    "with candidate scores, overall rating, and a short rationale for each. "
    "Choose the best overall response."
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
        overlap_score = min(10.0, overlap * 1.5)
        length_score = min(10.0, max(0.0, len(cand.text) / 25.0))
        coherence = min(10.0, 4.0 + length_score)
        scene_advancement = min(10.0, (overlap_score + length_score) / 2 + 3.0)
        emotional = min(10.0, 5.0 + length_score / 2)
        scores = {
            "In-Character Fidelity": float(overlap_score),
            "Continuity": float(min(10.0, overlap_score + 2.0)),
            "Emotional Realism": float(emotional),
            "Scene Advancement": float(scene_advancement),
            "Coherence": float(coherence),
        }
        overall = sum(scores.values()) / len(scores)
        judged.append(
            Judged(candidate=cand, scores=scores, overall=float(overall), rationale="Heuristic fallback")
        )
    return judged


JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_key(label: str) -> str:
    """Normalize a rubric label for comparison."""

    normalized = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", normalized.lower())


def _safe_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, str) and not value.strip():
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _compute_overall(scores: Dict[str, float], weights: JudgeRubricWeights) -> float:
    normalized_scores = {_normalize_key(key): value for key, value in scores.items()}
    weight_dict = { _normalize_key(key): weight for key, weight in weights.as_dict().items() }
    weighted_total = 0.0
    used_weight = 0.0
    for key, weight in weight_dict.items():
        if key in normalized_scores:
            weighted_total += (normalized_scores[key] / 10.0) * weight
            used_weight += weight
    if used_weight > 0:
        return round((weighted_total / used_weight) * 10.0, 4)
    if scores:
        return round(sum(scores.values()) / len(scores), 4)
    return 0.0


def _parse_judge_payload(text: str) -> Dict[str, Any]:
    try:
        return orjson.loads(text)
    except Exception:
        match = JSON_PATTERN.search(text)
        if not match:
            raise
        segment = match.group(0)
        try:
            return orjson.loads(segment)
        except Exception:
            return json.loads(segment)


def _coerce_scores(raw_scores: Any) -> Dict[str, float]:
    if isinstance(raw_scores, dict):
        normalized: Dict[str, float] = {}
        for key, value in raw_scores.items():
            score = _safe_float(value)
            if score is not None:
                normalized[str(key)] = float(score)
        return normalized
    if isinstance(raw_scores, list):
        normalized = {}
        for item in raw_scores:
            if not isinstance(item, dict):
                continue
            label = item.get("dimension") or item.get("name") or item.get("criterion") or item.get("label")
            score = item.get("score") or item.get("value") or item.get("rating")
            if label is None:
                continue
            number = _safe_float(score)
            if number is None:
                continue
            normalized[str(label)] = float(number)
        return normalized
    return {}


def _extract_candidate_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    preferred_keys = [
        "candidates",
        "candidate_scores",
        "candidateRatings",
        "ratings",
        "evaluations",
        "results",
    ]
    for key in preferred_keys:
        for actual_key, value in payload.items():
            if _normalize_key(actual_key) == _normalize_key(key) and isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    for value in payload.values():
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return list(value)
    return []


def _extract_overall(raw: Dict[str, Any], scores: Dict[str, float], weights: JudgeRubricWeights) -> float:
    for key in ("overall", "rating", "score", "overall_rating", "overallScore"):
        if key in raw:
            value = _safe_float(raw[key])
            if value is not None:
                return round(float(value), 4)
    normalized_scores = {_normalize_key(k): v for k, v in scores.items()}
    return _compute_overall(normalized_scores, weights)


def _parse_judge_payload(text: str) -> Dict[str, Any]:
    try:
        return orjson.loads(text)
    except Exception:
        match = JSON_PATTERN.search(text)
        if not match:
            raise
        segment = match.group(0)
        try:
            return orjson.loads(segment)
        except Exception:
            return json.loads(segment)


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
        payload = _parse_judge_payload(text)
        judged_slots: List[Optional[Judged]] = [None] * len(candidates)
        results = _extract_candidate_results(payload)
        indexed_results: Dict[int, Dict[str, Any]] = {}
        for result in results:
            candidate_index = None
            if "index" in result:
                idx_val = _safe_float(result["index"])
                if idx_val is not None:
                    candidate_index = int(idx_val)
            if candidate_index is None:
                for key in ("candidate", "candidate_id", "id"):
                    if key in result:
                        idx_val = _safe_float(result[key])
                        if idx_val is not None:
                            candidate_index = int(idx_val)
                            break
            if candidate_index is not None and 1 <= candidate_index <= len(candidates):
                indexed_results[candidate_index - 1] = result
        for idx, cand in enumerate(candidates):
            if idx < len(results):
                result = results[idx]
            elif idx in indexed_results:
                result = indexed_results[idx]
            else:
                result = {}
            scores = _coerce_scores(result.get("scores") or result.get("dimensions") or result.get("criteria"))
            if not scores and any(key in result for key in ("In-Character Fidelity", "Continuity", "Emotional Realism", "Scene Advancement", "Coherence")):
                scores = _coerce_scores({
                    key: result[key]
                    for key in ("In-Character Fidelity", "Continuity", "Emotional Realism", "Scene Advancement", "Coherence")
                    if key in result
                })
            if scores:
                overall_value = _extract_overall(result, scores, weights)
                judged_slots[idx] = Judged(
                    candidate=cand,
                    scores=scores,
                    overall=float(overall_value),
                    rationale=str(result.get("rationale", result.get("explanation", ""))),
                )
        fallback_used = False
        for idx, judged_item in enumerate(judged_slots):
            if judged_item is None:
                judged_slots[idx] = _fallback_scores(history, [candidates[idx]])[0]
                fallback_used = True
        judged = [item for item in judged_slots if item is not None]
        metadata.update({"usage": usage, "raw": raw})
        if fallback_used:
            metadata["fallback"] = True
            metadata["fallback_reason"] = "partial"
        return judged, metadata
    except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
        metadata.update({"fallback": True, "error": str(exc)})
        judged = _fallback_scores(history, candidates)
        return judged, metadata


__all__ = ["score_candidates"]
