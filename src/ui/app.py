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
    "GEN_TIMEOUT_S": "60",
    "JUDGE_TEMPERATURE": "0.0",
    "JUDGE_TOP_P": "1.0",
    "JUDGE_MAX_TOKENS": "800",
    "JUDGE_PRESENCE_PENALTY": "0.0",
    "JUDGE_FREQUENCY_PENALTY": "0.0",
    "JUDGE_TIMEOUT_S": "60",
    "CONTEXT_K": "8",
    "ENABLE_MEMORY": "true",
    "GEN_SYSTEM_PROMPT": DEFAULT_GEN_PROMPT,
    "JUDGE_SYSTEM_PROMPT": DEFAULT_JUDGE_PROMPT,
}

def _decode_env_value(value: str) -> str:
    """Decode escaped characters stored in .env values."""

    if not value:
        return value

    decoded = value.replace("\\r\\n", "\n").replace("\\r", "\n").replace("\\n", "\n")
    decoded = decoded.replace("\\\\", "\\")
    return decoded


def load_env_defaults() -> Dict[str, str]:
    values = DEFAULTS.copy()
    env_values = {
        k.upper(): _decode_env_value(v)
        for k, v in dotenv_values(".env").items()
        if v is not None
    }
    values.update(env_values)
    legacy_timeout = values.get("TIMEOUT_S")
    if "GEN_TIMEOUT_S" not in values and legacy_timeout is not None:
        values["GEN_TIMEOUT_S"] = legacy_timeout
    if "JUDGE_TIMEOUT_S" not in values and legacy_timeout is not None:
        values["JUDGE_TIMEOUT_S"] = legacy_timeout
    return values


def init_state() -> None:
    if "dialogue" not in st.session_state:
        st.session_state.dialogue = []  # type: ignore[assignment]
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager()
    if "last_judged" not in st.session_state:
        st.session_state.last_judged = []  # type: ignore[assignment]
    if "show_candidates" not in st.session_state:
        st.session_state.show_candidates = False
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {}
    if "conversation_title" not in st.session_state:
        st.session_state.conversation_title = "New Conversation"
    if "title_generated" not in st.session_state:
        st.session_state.title_generated = False
    if "pending_title_prompt" not in st.session_state:
        st.session_state.pending_title_prompt = ""
    if "settings" not in st.session_state:
        st.session_state.settings = load_env_defaults()


def new_chat() -> None:
    st.session_state.dialogue = []
    st.session_state.memory_manager = MemoryManager()
    st.session_state.last_judged = []
    st.session_state.token_usage = {}
    st.session_state.conversation_title = "New Conversation"
    st.session_state.title_generated = False
    st.session_state.pending_title_prompt = ""


async def run_pipeline(
    settings: Dict[str, Any],
    history: Sequence[Turn],
    memory_manager: MemoryManager,
) -> Tuple[
    Dict[str, Any],
    Sequence[Candidate],
    Sequence[Judged],
    Judged,
    Dict[str, Any],
]:
    history_list: List[Turn] = list(history)
    state = DialogueState(history=history_list, memory=memory_manager.store)
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
        timeout=float(settings["GEN_TIMEOUT_S"]),
        system_prompt=str(settings.get("GEN_SYSTEM_PROMPT", "")),
    )
    judged, judge_meta = await score_candidates(
        base_url=settings["JUDGE_BASE_URL"],
        api_key=settings["JUDGE_API_KEY"],
        model=settings["JUDGE_MODEL"],
        history=recent_turns,
        candidates=candidates,
        weights=JudgeRubricWeights(),
        temperature=float(settings["JUDGE_TEMPERATURE"]),
        top_p=float(settings["JUDGE_TOP_P"]),
        max_tokens=int(settings["JUDGE_MAX_TOKENS"]),
        presence_penalty=float(settings["JUDGE_PRESENCE_PENALTY"]),
        frequency_penalty=float(settings["JUDGE_FREQUENCY_PENALTY"]),
        timeout=float(settings["JUDGE_TIMEOUT_S"]),
        system_prompt=str(settings.get("JUDGE_SYSTEM_PROMPT", "")),
    )
    best = select_best(judged)
    return context_pack, candidates, judged, best, judge_meta


def _apply_generation_success(
    context_pack: Dict[str, Any],
    candidates: Sequence[Candidate],
    judged: Sequence[Judged],
    best: Judged,
    judge_meta: Dict[str, Any],
) -> None:
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


async def _request_conversation_title(
    settings: Dict[str, Any],
    first_message: str,
) -> str:
    prompt = (
        "You name chat conversations. Read the first user message and reply with a concise "
        "title of no more than three words. Use title case."
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"First message: {first_message}\nRespond with only the title.",
        },
    ]
    text, _usage, _raw = await chat_completion(
        base_url=settings["GEN_BASE_URL"],
        api_key=settings["GEN_API_KEY"],
        model=settings["GEN_MODEL"],
        messages=messages,
        temperature=min(1.0, float(settings.get("TEMPERATURE", 0.7))),
        top_p=float(settings.get("TOP_P", 0.9)),
        max_tokens=12,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        timeout=float(settings.get("GEN_TIMEOUT_S", 60)),
    )
    return text.strip()


def _shorten_title(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "Conversation"
    words = cleaned.split()
    return " ".join(words[:3])[:48]


def _update_conversation_title_if_needed(settings: Dict[str, Any]) -> None:
    if st.session_state.title_generated:
        return
    first_message = st.session_state.pending_title_prompt.strip()
    if not first_message:
        return
    try:
        raw_title = asyncio.run(_request_conversation_title(settings, first_message))
        candidate = _shorten_title(raw_title)
    except Exception:  # pragma: no cover - depends on external API
        candidate = _shorten_title(first_message)
    st.session_state.conversation_title = candidate
    st.session_state.title_generated = True
    st.session_state.pending_title_prompt = ""


def _execute_generation(settings: Dict[str, Any], *, new_user: bool) -> None:
    history_snapshot = [Turn(**turn.dict()) for turn in st.session_state.dialogue]
    memory_manager: MemoryManager = st.session_state.memory_manager
    try:
        context_pack, candidates, judged, best, judge_meta = asyncio.run(
            run_pipeline(settings, history_snapshot, memory_manager)
        )
    except Exception as exc:  # pragma: no cover - network error path
        st.error(f"Generation failed: {exc}")
        if new_user and st.session_state.dialogue and st.session_state.dialogue[-1].role == "user":
            st.session_state.dialogue.pop()
            st.session_state.pending_title_prompt = ""
        st.session_state.last_judged = []
        st.session_state.token_usage = {}
        return

    _apply_generation_success(context_pack, candidates, judged, best, judge_meta)
    if settings:
        _update_conversation_title_if_needed(settings)


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


def handle_send(user_text: str) -> bool:
    message = user_text.strip()
    if not message:
        return False
    st.session_state.dialogue.append(Turn(role="user", content=message))
    if not st.session_state.title_generated and not st.session_state.pending_title_prompt:
        st.session_state.pending_title_prompt = message
    return True


def handle_regenerate() -> bool:
    if not st.session_state.dialogue or st.session_state.dialogue[-1].role != "assistant":
        return False
    st.session_state.dialogue.pop()
    return True


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
        with st.expander("Generator decoding"):
            settings["TEMPERATURE"] = st.slider("Temperature", 0.0, 2.0, float(settings["TEMPERATURE"]))
            settings["TOP_P"] = st.slider("Top-p", 0.1, 1.0, float(settings["TOP_P"]))
            settings["MAX_TOKENS"] = st.slider("Max tokens", 64, 2048, int(settings["MAX_TOKENS"]))
            settings["N_CANDIDATES"] = st.slider("Candidates", 3, 5, int(settings["N_CANDIDATES"]))
            settings["PRESENCE_PENALTY"] = st.slider(
                "Presence penalty", -2.0, 2.0, float(settings["PRESENCE_PENALTY"])
            )
            settings["FREQUENCY_PENALTY"] = st.slider(
                "Frequency penalty", -2.0, 2.0, float(settings["FREQUENCY_PENALTY"])
            )
            settings["GEN_TIMEOUT_S"] = st.slider(
                "Timeout (s)", 10, 120, int(settings["GEN_TIMEOUT_S"]), help="Generator request timeout"
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
        with st.expander("Judge decoding"):
            settings["JUDGE_TEMPERATURE"] = st.slider(
                "Judge temperature", 0.0, 2.0, float(settings["JUDGE_TEMPERATURE"])
            )
            settings["JUDGE_TOP_P"] = st.slider(
                "Judge top-p", 0.1, 1.0, float(settings["JUDGE_TOP_P"])
            )
            settings["JUDGE_MAX_TOKENS"] = st.slider(
                "Judge max tokens", 128, 2048, int(settings["JUDGE_MAX_TOKENS"])
            )
            settings["JUDGE_PRESENCE_PENALTY"] = st.slider(
                "Judge presence penalty", -2.0, 2.0, float(settings["JUDGE_PRESENCE_PENALTY"])
            )
            settings["JUDGE_FREQUENCY_PENALTY"] = st.slider(
                "Judge frequency penalty", -2.0, 2.0, float(settings["JUDGE_FREQUENCY_PENALTY"])
            )
            settings["JUDGE_TIMEOUT_S"] = st.slider(
                "Judge timeout (s)", 10, 180, int(settings["JUDGE_TIMEOUT_S"])
            )
        if st.button("Test judge connection"):
            test_connection("Judge", settings["JUDGE_BASE_URL"], settings["JUDGE_API_KEY"], settings["JUDGE_MODEL"])
        st.subheader("Conversation")
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
                "GEN_TIMEOUT_S": settings["GEN_TIMEOUT_S"],
                "JUDGE_TEMPERATURE": settings["JUDGE_TEMPERATURE"],
                "JUDGE_TOP_P": settings["JUDGE_TOP_P"],
                "JUDGE_MAX_TOKENS": settings["JUDGE_MAX_TOKENS"],
                "JUDGE_PRESENCE_PENALTY": settings["JUDGE_PRESENCE_PENALTY"],
                "JUDGE_FREQUENCY_PENALTY": settings["JUDGE_FREQUENCY_PENALTY"],
                "JUDGE_TIMEOUT_S": settings["JUDGE_TIMEOUT_S"],
                "CONTEXT_K": settings["CONTEXT_K"],
            })
            st.success("Defaults saved to .env")
    return settings


def render_chat_area(settings: Dict[str, Any]) -> None:
    title = st.session_state.conversation_title or "Conversation"
    st.markdown(
        f"<div class='chat-header'><h2>{title}</h2></div>",
        unsafe_allow_html=True,
    )
    token_info = st.session_state.token_usage
    if token_info:
        st.markdown(
            f"<div class='token-counter'>Usage · Generator: {token_info.get('generator')} · Judge: {token_info.get('judge')}</div>",
            unsafe_allow_html=True,
        )

    chat_placeholder = st.empty()

    def draw_chat(thinking: bool = False) -> None:
        with chat_placeholder.container():
            inject_css()
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for idx, turn in enumerate(st.session_state.dialogue):
                render_message(turn, idx)
            if thinking:
                st.markdown(
                    """
                    <div class='chat-row right'>
                        <div class='chat-bubble assistant thinking-bubble'>
                            <div class='chat-role'>Assistant</div>
                            <div class='chat-content'><em>Thinking…</em></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
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

    action_taken = False

    if send_clicked:
        if handle_send(user_text):
            st.session_state.last_judged = []
            st.session_state.token_usage = {}
            draw_chat(thinking=True)
            with st.spinner("Generating assistant reply…"):
                _execute_generation(settings, new_user=True)
            draw_chat()
            action_taken = True
        else:
            st.warning("Please enter a message before sending.")
    elif regenerate_clicked:
        if handle_regenerate():
            st.session_state.last_judged = []
            st.session_state.token_usage = {}
            draw_chat(thinking=True)
            with st.spinner("Regenerating assistant reply…"):
                _execute_generation(settings, new_user=False)
            draw_chat()
            action_taken = True
        else:
            st.info("Nothing to regenerate yet.")
    elif stop_clicked:
        st.info("No generation in progress to stop.")

    if not action_taken:
        draw_chat()

    if st.session_state.show_candidates and st.session_state.last_judged:
        st.markdown("### Candidate Rankings")
        render_candidates(st.session_state.last_judged)


def main() -> None:
    init_state()
    settings = st.session_state.settings
    st.session_state.settings = render_sidebar(settings)
    render_chat_area(st.session_state.settings)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
