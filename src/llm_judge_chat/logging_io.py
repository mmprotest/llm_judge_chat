"""Logging and dataset export utilities."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

try:  # pragma: no cover - fallback when orjson missing
    import orjson
except ImportError:  # pragma: no cover
    import json as _json

    class _OrjsonFallback:
        @staticmethod
        def dumps(obj: Dict[str, object]) -> bytes:
            return _json.dumps(obj).encode("utf-8")

    orjson = _OrjsonFallback()  # type: ignore

from .schemas import Candidate, Judged
from .utils import append_jsonl, load_json, timestamp

LOG_DIR = Path("logs")


def _session_path() -> Path:
    today = datetime.utcnow().strftime("%Y%m%d")
    return LOG_DIR / f"session_{today}.jsonl"


def log_turn(
    *,
    context_pack: Dict[str, object],
    candidates: Sequence[Candidate],
    judged: Sequence[Judged],
    chosen: str,
    usage: Dict[str, object] | None,
    metadata: Dict[str, object] | None = None,
) -> Path:
    """Append a dialogue turn to the current session log."""

    record = {
        "timestamp": timestamp(),
        "context_pack": context_pack,
        "candidates": [cand.dict() for cand in candidates],
        "judge": [item.dict() for item in judged],
        "chosen": chosen,
        "usage": usage,
        "metadata": metadata or {},
    }
    path = _session_path()
    append_jsonl(path, record)
    return path


def export_pairs_for_dpo(logs_dir: str, output_path: str, min_gap: float = 0.5) -> None:
    """Export preference pairs for DPO training."""

    logs_path = Path(logs_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as fh:
        for log_file in sorted(logs_path.glob("session_*.jsonl")):
            for record in load_json(log_file):
                judged = record.get("judge", [])
                if len(judged) < 2:
                    continue
                sorted_candidates = sorted(judged, key=lambda item: item["overall"], reverse=True)
                best = sorted_candidates[0]
                others = sorted_candidates[1:]
                losers = [cand for cand in others if best["overall"] - cand["overall"] >= min_gap]
                if not losers:
                    continue
                rejected = random.choice(losers)
                context = record.get("context_pack", {})
                sample = {
                    "context": context,
                    "chosen": best["candidate"]["text"],
                    "rejected": rejected["candidate"]["text"],
                }
                fh.write(orjson.dumps(sample))
                fh.write(b"\n")


__all__ = ["log_turn", "export_pairs_for_dpo"]
