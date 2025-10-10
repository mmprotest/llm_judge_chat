"""Streamlit UI for the LLM judge chat application."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence, Tuple

import streamlit as st
from dotenv import dotenv_values

from llm_judge_chat.context import build_context, window_history
from llm_judge_chat.generator import SYSTEM_PROMPT as DEFAULT_GEN_PROMPT, generate_candidates
from llm_judge_chat.judge import JUDGE_PROMPT as DEFAULT_JUDGE_PROMPT, score_candidates
from llm_judge_chat.logging_io import log_turn
from llm_judge_chat.memory import MemoryManager
from llm_judge_chat.openai_compat import chat_completion
from llm_judge_chat.schemas import Candidate, DialogueState, JudgeRubricWeights, Judged, Turn
from llm_judge_chat.selector import select_best
from llm_judge_chat.utils import estimate_tokens
from llm_judge_chat.config import persist_settings
try:  # pragma: no cover - import path differs when run via ``streamlit run``
    from .components import inject_css, render_candidates, render_message
except ImportError:  # pragma: no cover - executed when module has no package context
    from components import inject_css, render_candidates, render_message

st.set_page_config(page_title="LLM Judge Chat", layout="wide")

DEFAULTS = {
    "GEN_BASE_URL": "https://api.openai.com/v1",
    "GEN_API_KEY": "",
    "GEN_MODEL": "gpt-3.5-turbo",
    "JUDGE_BASE_URL": "https://api.openai.com/v1",
    "JUDGE_API_KEY": "",
    "JUDGE_MODEL": "gpt-4",
    "TEMPERATURE": "0.8",
    "TOP_P": "0.9",
    "MAX_TOKENS": "512",
    "N_CANDIDATES": "3",
    "PRESENCE_PENALTY": "0.0",
    "FREQUENCY_PENALTY": "0.0",
    "TIMEOUT_S": "60",
    "CONTEXT_K": "8",
    "ENABLE_MEMORY": "true",
    "GEN_SYSTEM_PROMPT": DEFAULT_GEN_PROMPT,
    "JUDGE_SYSTEM_PROMPT": DEFAULT_JUDGE_PROMPT,
}


def load_env_defaults() -> Dict[str, str]:
    values = DEFAULTS.copy()
    values.update({k.upper(): v for k, v in dotenv_values(".env").items() if v is not None})
    return values


def init_state() -> None:
    if "dialogue" not in st.session_state:
        st.session_state.dialogue = []  # type: ignore[assignment]
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager()
    if "running" not in st.session_state:
        st.session_state.running = False
    if "last_judged" not in st.session_state:
        st.session_state.last_judged = []  # type: ignore[assignment]
    if "show_candidates" not in st.session_state:
        st.session_state.show_candidates = False
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {}


def new_chat() -> None:
    st.session_state.dialogue = []
    st.session_state.memory_manager = MemoryManager()
    st.session_state.last_judged = []
    st.session_state.token_usage = {}


async def run_pipeline(settings: Dict[str, Any]) -> Tuple[
    Dict[str, Any],
    Sequence[Candidate],
    Sequence[Judged],
    Judged,
    Dict[str, Any],
]:
    memory_manager: MemoryManager = st.session_state.memory_manager
    history: List[Turn] = st.session_state.dialogue  # type: ignore[assignment]
    state = DialogueState(history=history, memory=memory_manager.store)
    context_pack = build_context(state, memory_manager, k=int(settings["CONTEXT_K"]))
    recent_turns = window_history(state.history, k=int(settings["CONTEXT_K"]))
    candidates = await generate_candidates(
        base_url=settings["GEN_BASE_URL"],
        api_key=settings["GEN_API_KEY"],
        model=settings["GEN_MODEL"],
        history=recent_turns,
        memory_summary=context_pack.get("memory_summary", {}),
        n=int(settings["N_CANDIDATES"]),
        temperature=float(settings["TEMPERATURE"]),
        top_p=float(settings["TOP_P"]),
        max_tokens=int(settings["MAX_TOKENS"]),
        presence_penalty=float(settings["PRESENCE_PENALTY"]),
        frequency_penalty=float(settings["FREQUENCY_PENALTY"]),
        timeout=float(settings["TIMEOUT_S"]),
        system_prompt=str(settings.get("GEN_SYSTEM_PROMPT", "")),
    )
    judged, judge_meta = await score_candidates(
        base_url=settings["JUDGE_BASE_URL"],
        api_key=settings["JUDGE_API_KEY"],
        model=settings["JUDGE_MODEL"],
        history=recent_turns,
        candidates=candidates,
        weights=JudgeRubricWeights(),
        timeout=float(settings["TIMEOUT_S"]),
        system_prompt=str(settings.get("JUDGE_SYSTEM_PROMPT", "")),
    )
    best = select_best(judged)
    return context_pack, candidates, judged, best, judge_meta


def execute_generation(settings: Dict[str, Any], *, new_user: bool) -> None:
    try:
        context_pack, candidates, judged, best, judge_meta = asyncio.run(run_pipeline(settings))
    except Exception as exc:  # pragma: no cover - network error path
        st.error(f"Generation failed: {exc}")
        if new_user and st.session_state.dialogue:
            st.session_state.dialogue.pop()
        st.session_state.running = False
        return

    st.session_state.dialogue.append(
        Turn(
            role="assistant",
            content=best.candidate.text,
            meta={"style": best.candidate.meta.get("decoding", {}).get("style")},
        )
    )
    st.session_state.last_judged = list(judged)
    usage_summary = {
        "generator": [cand.meta.get("usage") for cand in candidates if cand.meta.get("usage")],
        "judge": judge_meta.get("usage"),
    }
    display_usage = {
        "generator": _format_usage(usage_summary["generator"], best.candidate.text),
        "judge": _format_usage([usage_summary["judge"]] if usage_summary["judge"] else [], ""),
    }
    st.session_state.token_usage = display_usage
    metadata = {k: v for k, v in judge_meta.items() if k != "usage"}
    log_turn(
        context_pack=context_pack,
        candidates=candidates,
        judged=judged,
        chosen=best.candidate.text,
        usage=usage_summary,
        metadata=metadata,
    )
    st.session_state.running = False
    st.experimental_rerun()


def _format_usage(usages: Sequence[Dict[str, Any]], text: str) -> str:
    if usages:
        usage = usages[0]
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if prompt is not None and completion is not None:
            return f"prompt {prompt} · completion {completion}"
    if text:
        estimate = estimate_tokens(text)
        return f"≈{estimate} tokens"
    return "n/a"


def handle_send(user_text: str, settings: Dict[str, Any]) -> None:
    if not user_text.strip():
        return
    if st.session_state.running:
        st.warning("Generation already in progress.")
        return
    st.session_state.running = True
    st.session_state.dialogue.append(Turn(role="user", content=user_text.strip()))
    execute_generation(settings, new_user=True)


def handle_regenerate(settings: Dict[str, Any]) -> None:
    if st.session_state.running:
        return
    if not st.session_state.dialogue or st.session_state.dialogue[-1].role != "assistant":
        return
    st.session_state.dialogue.pop()
    st.session_state.running = True
    execute_generation(settings, new_user=False)


def test_connection(name: str, base_url: str, api_key: str, model: str) -> None:
    try:
        asyncio.run(
            chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=[{"role": "system", "content": "Ping"}, {"role": "user", "content": "Respond with ok"}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                timeout=10.0,
            )
        )
        st.success(f"{name} connection OK")
    except Exception as exc:  # pragma: no cover - network path
        st.error(f"{name} connection failed: {exc}")


def render_sidebar(settings: Dict[str, Any]) -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("<button class='new-chat-button'>New Chat</button>", unsafe_allow_html=True)
        if st.button("Reset conversation"):
            new_chat()
            st.experimental_rerun()
        st.subheader("Generator")
        settings["GEN_BASE_URL"] = st.text_input("Base URL", value=settings["GEN_BASE_URL"])
        settings["GEN_API_KEY"] = st.text_input("API Key", value=settings["GEN_API_KEY"], type="password")
        settings["GEN_MODEL"] = st.text_input("Model", value=settings["GEN_MODEL"])
        settings["GEN_SYSTEM_PROMPT"] = st.text_area(
            "System prompt",
            value=settings["GEN_SYSTEM_PROMPT"],
            height=120,
            help="Customize the assistant's behavior and tone.",
        )
        if st.button("Test generator connection"):
            test_connection("Generator", settings["GEN_BASE_URL"], settings["GEN_API_KEY"], settings["GEN_MODEL"])
        st.subheader("Judge")
        settings["JUDGE_BASE_URL"] = st.text_input("Judge Base URL", value=settings["JUDGE_BASE_URL"])
        settings["JUDGE_API_KEY"] = st.text_input("Judge API Key", value=settings["JUDGE_API_KEY"], type="password")
        settings["JUDGE_MODEL"] = st.text_input("Judge Model", value=settings["JUDGE_MODEL"])
        settings["JUDGE_SYSTEM_PROMPT"] = st.text_area(
            "Judge prompt",
            value=settings["JUDGE_SYSTEM_PROMPT"],
            height=120,
            help="Add evaluation goals or weighting guidance for the judge.",
        )
        if st.button("Test judge connection"):
            test_connection("Judge", settings["JUDGE_BASE_URL"], settings["JUDGE_API_KEY"], settings["JUDGE_MODEL"])
        st.subheader("Decoding")
        settings["TEMPERATURE"] = st.slider("Temperature", 0.0, 2.0, float(settings["TEMPERATURE"]))
        settings["TOP_P"] = st.slider("Top-p", 0.1, 1.0, float(settings["TOP_P"]))
        settings["MAX_TOKENS"] = st.slider("Max tokens", 64, 2048, int(settings["MAX_TOKENS"]))
        settings["N_CANDIDATES"] = st.slider("Candidates", 3, 5, int(settings["N_CANDIDATES"]))
        settings["PRESENCE_PENALTY"] = st.slider("Presence penalty", -2.0, 2.0, float(settings["PRESENCE_PENALTY"]))
        settings["FREQUENCY_PENALTY"] = st.slider("Frequency penalty", -2.0, 2.0, float(settings["FREQUENCY_PENALTY"]))
        with st.expander("Advanced"):
            settings["TIMEOUT_S"] = st.slider("Timeout (s)", 10, 120, int(settings["TIMEOUT_S"]))
            settings["CONTEXT_K"] = st.slider("Context turns", 2, 16, int(settings["CONTEXT_K"]))
        if st.button("Save as defaults"):
            persist_settings({
                "GEN_BASE_URL": settings["GEN_BASE_URL"],
                "GEN_MODEL": settings["GEN_MODEL"],
                "GEN_SYSTEM_PROMPT": settings["GEN_SYSTEM_PROMPT"],
                "JUDGE_BASE_URL": settings["JUDGE_BASE_URL"],
                "JUDGE_MODEL": settings["JUDGE_MODEL"],
                "JUDGE_SYSTEM_PROMPT": settings["JUDGE_SYSTEM_PROMPT"],
                "TEMPERATURE": settings["TEMPERATURE"],
                "TOP_P": settings["TOP_P"],
                "MAX_TOKENS": settings["MAX_TOKENS"],
                "N_CANDIDATES": settings["N_CANDIDATES"],
                "PRESENCE_PENALTY": settings["PRESENCE_PENALTY"],
                "FREQUENCY_PENALTY": settings["FREQUENCY_PENALTY"],
                "TIMEOUT_S": settings["TIMEOUT_S"],
                "CONTEXT_K": settings["CONTEXT_K"],
            })
            st.success("Defaults saved to .env")
    return settings


def render_chat_area(settings: Dict[str, Any]) -> None:
    st.markdown(
        f"<div class='chat-header'><h2>{settings['GEN_MODEL']}</h2></div>",
        unsafe_allow_html=True,
    )
    token_info = st.session_state.token_usage
    if token_info:
        st.markdown(
            f"<div class='token-counter'>Usage · Generator: {token_info.get('generator')} · Judge: {token_info.get('judge')}</div>",
            unsafe_allow_html=True,
        )
    chat_container = st.container()
    with chat_container:
        inject_css()
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for idx, turn in enumerate(st.session_state.dialogue):
            render_message(turn, idx)
        st.markdown("</div>", unsafe_allow_html=True)
    with st.form("chat-input", clear_on_submit=True):
        user_text = st.text_area("Message", height=120, key="chat_input")
        cols = st.columns([1, 1, 1, 2])
        send_clicked = cols[0].form_submit_button("Send", type="primary")
        regenerate_clicked = cols[1].form_submit_button("Regenerate")
        stop_clicked = cols[2].form_submit_button("Stop")
        st.session_state.show_candidates = cols[3].checkbox(
            "Show candidates & scores", value=st.session_state.show_candidates
        )
    if send_clicked:
        handle_send(user_text, settings)
    elif regenerate_clicked:
        handle_regenerate(settings)
    elif stop_clicked:
        st.session_state.running = False
    if st.session_state.show_candidates and st.session_state.last_judged:
        st.markdown("### Candidate Rankings")
        render_candidates(st.session_state.last_judged)


def main() -> None:
    init_state()
    settings = load_env_defaults()
    settings = render_sidebar(settings)
    render_chat_area(settings)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
