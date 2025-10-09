from __future__ import annotations

import json
import random
from pathlib import Path

import json

from llm_judge_chat.logging_io import export_pairs_for_dpo
from llm_judge_chat.schemas import Candidate, Judged
from llm_judge_chat.selector import select_best


def test_select_best() -> None:
    candidates = [
        Judged(candidate=Candidate(text="a", meta={}), scores={}, overall=0.5, rationale=""),
        Judged(candidate=Candidate(text="b", meta={}), scores={}, overall=0.9, rationale=""),
        Judged(candidate=Candidate(text="c", meta={}), scores={}, overall=0.1, rationale=""),
    ]
    best = select_best(candidates)
    assert best.candidate.text == "b"


def test_export_pairs_for_dpo(tmp_path: Path, monkeypatch) -> None:
    random.seed(0)
    sample = {
        "timestamp": "2024-01-01T00:00:00Z",
        "context_pack": {"history": [], "memory_summary": {}},
        "candidates": [],
        "judge": [
            {
                "candidate": {"text": "best"},
                "scores": {"relevance": 9.0},
                "overall": 9.0,
                "rationale": "",
            },
            {
                "candidate": {"text": "mid"},
                "scores": {"relevance": 6.0},
                "overall": 6.0,
                "rationale": "",
            },
            {
                "candidate": {"text": "low"},
                "scores": {"relevance": 2.0},
                "overall": 2.0,
                "rationale": "",
            },
        ],
        "chosen": "best",
        "usage": {},
    }
    log_file = tmp_path / "session_20240101.jsonl"
    with log_file.open("wb") as fh:
        fh.write(json.dumps(sample).encode("utf-8"))
        fh.write(b"\n")

    output = tmp_path / "pairs.jsonl"
    export_pairs_for_dpo(str(tmp_path), str(output), min_gap=0.5)

    lines = output.read_text().strip().splitlines()
    assert lines, "Expected at least one pair"
    pair = json.loads(lines[0])
    assert pair["chosen"] == "best"
    assert pair["rejected"] in {"mid", "low"}
