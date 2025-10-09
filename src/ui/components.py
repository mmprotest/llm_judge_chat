"""Reusable UI components for the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import streamlit as st

from llm_judge_chat.schemas import Judged, Turn

CSS_PATH = Path(__file__).with_name("styles.css")


def inject_css() -> None:
    """Inject custom CSS styling."""

    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_message(turn: Turn, index: int) -> None:
    """Render a single chat message bubble."""

    align = "left" if turn.role == "user" else "right"
    role_class = "user" if turn.role == "user" else "assistant"
    bubble = f"""
    <div class="chat-row {align}">
        <div class="chat-bubble {role_class}">
            <div class="chat-role">{turn.role.title()}</div>
            <div class="chat-content">{turn.content}</div>
        </div>
    </div>
    """
    st.markdown(bubble, unsafe_allow_html=True)


def render_candidates(judged: Sequence[Judged]) -> None:
    """Render candidate cards with scores and rationale."""

    for rank, item in enumerate(sorted(judged, key=lambda i: i.overall, reverse=True), start=1):
        score_rows = "".join(
            f"<div class='score-item'><span>{name.title()}</span><span>{value:.1f}</span></div>"
            for name, value in item.scores.items()
        )
        card = f"""
        <div class="candidate-card">
            <div class="candidate-rank">#{rank} · {item.overall:.2f}</div>
            <div class="candidate-text">{item.candidate.text}</div>
            <div class="candidate-scores">{score_rows}</div>
            <div class="candidate-rationale">{item.rationale}</div>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)


__all__ = ["inject_css", "render_message", "render_candidates"]
