"""Streamlit UI for the LLM judge chat application."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _trigger_rerun() -> None:
    try:
        st.experimental_rerun()
    except AttributeError:  # pragma: no cover - depends on Streamlit version
        st.rerun()


def _force_rerun() -> None:
    _trigger_rerun()
    try:
        st.stop()
    except Exception:  # pragma: no cover - safety for test execution
        pass


def _decode_env_value(value: str) -> str:
    """Decode escaped characters stored in .env values."""

    if not value:
        return value

    decoded = value.replace("\\r\\n", "\n").replace("\\r", "\n").replace("\\n", "\n")
    decoded = decoded.replace("\\\\", "\\")
    return decoded


def _trigger_rerun() -> None:
    try:
        st.experimental_rerun()
    except AttributeError:  # pragma: no cover - depends on Streamlit version
        st.rerun()


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
    if "multi_dialogue" not in st.session_state:
        st.session_state.multi_dialogue = []  # type: ignore[assignment]
    if "multi_memory_manager" not in st.session_state:
        st.session_state.multi_memory_manager = MemoryManager()
    if "multi_candidates" not in st.session_state:
        st.session_state.multi_candidates = []  # type: ignore[assignment]
    if "multi_context_pack" not in st.session_state:
        st.session_state.multi_context_pack = {}
    if "multi_token_usage" not in st.session_state:
        st.session_state.multi_token_usage = {}
    if "multi_pending_generation" not in st.session_state:
        st.session_state.multi_pending_generation = None
    if "multi_is_thinking" not in st.session_state:
        st.session_state.multi_is_thinking = False
    if "multi_editing_index" not in st.session_state:
        st.session_state.multi_editing_index = None  # type: ignore[assignment]
    if "multi_editing_text" not in st.session_state:
        st.session_state.multi_editing_text = ""
    if "multi_edit_notice" not in st.session_state:
        st.session_state.multi_edit_notice = ""
    if "conversation_title" not in st.session_state:
        st.session_state.conversation_title = "New Conversation"
    if "title_generated" not in st.session_state:
        st.session_state.title_generated = False
    if "pending_title_prompt" not in st.session_state:
        st.session_state.pending_title_prompt = ""
    if "settings" not in st.session_state:
        st.session_state.settings = load_env_defaults()
    if "editing_index" not in st.session_state:
        st.session_state.editing_index = None  # type: ignore[assignment]
    if "editing_text" not in st.session_state:
        st.session_state.editing_text = ""
    if "edit_notice" not in st.session_state:
        st.session_state.edit_notice = ""
    if "pending_generation" not in st.session_state:
        st.session_state.pending_generation = None
    if "is_thinking" not in st.session_state:
        st.session_state.is_thinking = False


def new_chat() -> None:
    st.session_state.dialogue = []
    st.session_state.memory_manager = MemoryManager()
    st.session_state.last_judged = []
    st.session_state.token_usage = {}
    st.session_state.multi_dialogue = []
    st.session_state.multi_memory_manager = MemoryManager()
    st.session_state.multi_candidates = []
    st.session_state.multi_context_pack = {}
    st.session_state.multi_token_usage = {}
    st.session_state.multi_pending_generation = None
    st.session_state.multi_is_thinking = False
    st.session_state.multi_editing_index = None
    st.session_state.multi_editing_text = ""
    st.session_state.multi_edit_notice = ""
    st.session_state.conversation_title = "New Conversation"
    st.session_state.title_generated = False
    st.session_state.pending_title_prompt = ""
    st.session_state.editing_index = None
    st.session_state.editing_text = ""
    st.session_state.edit_notice = ""
    st.session_state.pending_generation = None
    st.session_state.is_thinking = False


def _rebuild_memory(history: Sequence[Turn]) -> MemoryManager:
    manager = MemoryManager()
    turns = list(history)
    for end in range(len(turns)):
        manager.update(turns[: end + 1])
    return manager


def _reset_title_seed(*, force: bool = False) -> None:
    first_user: Optional[str] = None
    for turn in st.session_state.dialogue:
        if turn.role == "user":
            first_user = turn.content
            break
    if first_user:
        st.session_state.pending_title_prompt = first_user
        if force or not st.session_state.title_generated:
            st.session_state.title_generated = False
            st.session_state.conversation_title = "New Conversation"
    else:
        st.session_state.pending_title_prompt = ""
        st.session_state.title_generated = False
        st.session_state.conversation_title = "New Conversation"


def _clear_edit_state() -> None:
    index = st.session_state.editing_index
    if index is not None:
        st.session_state.pop(f"edit_area_{index}", None)
    st.session_state.editing_index = None
    st.session_state.editing_text = ""


def _clear_multi_edit_state() -> None:
    index = st.session_state.multi_editing_index
    if index is not None:
        st.session_state.pop(f"multi_edit_area_{index}", None)
    st.session_state.multi_editing_index = None
    st.session_state.multi_editing_text = ""


def _begin_edit(index: int) -> None:
    previous = st.session_state.editing_index
    if previous is not None and previous != index:
        st.session_state.pop(f"edit_area_{previous}", None)
    st.session_state.editing_index = index
    current = st.session_state.dialogue[index].content
    st.session_state.editing_text = current
    st.session_state.edit_notice = ""


def _begin_multi_edit(index: int) -> None:
    previous = st.session_state.multi_editing_index
    if previous is not None and previous != index:
        st.session_state.pop(f"multi_edit_area_{previous}", None)
    st.session_state.multi_editing_index = index
    current = st.session_state.multi_dialogue[index].content
    st.session_state.multi_editing_text = current
    st.session_state.multi_edit_notice = ""


def _editor_height(text: str) -> int:
    """Pick an editor height that roughly matches the current text."""

    lines = text.count("\n") + 1
    return max(120, min(280, lines * 26))


def _render_inline_editor(turn: Turn, idx: int) -> Tuple[str, bool, bool]:
    align = "left" if turn.role == "user" else "right"
    role_class = "user" if turn.role == "user" else "assistant"
    current_text = st.session_state.get(
        f"edit_area_{idx}", st.session_state.editing_text or turn.content
    )
    st.markdown(
        f"<div class='chat-row {align}'><div class='chat-bubble {role_class} editing' data-role='{role_class}'>",
        unsafe_allow_html=True,
    )
    edit_value = st.text_area(
        "Edit message",
        value=current_text,
        key=f"edit_area_{idx}",
        height=_editor_height(current_text),
        label_visibility="collapsed",
    )
    action_cols = st.columns([1, 1, 6])
    with action_cols[0]:
        save_clicked = st.button(
            "Save",
            key=f"save-edit-{idx}",
            use_container_width=True,
        )
    with action_cols[1]:
        cancel_clicked = st.button(
            "Cancel",
            key=f"cancel-edit-{idx}",
            use_container_width=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)
    return edit_value, save_clicked, cancel_clicked


def _render_multi_inline_editor(turn: Turn, idx: int) -> Tuple[str, bool, bool]:
    align = "left" if turn.role == "user" else "right"
    role_class = "user" if turn.role == "user" else "assistant"
    current_text = st.session_state.get(
        f"multi_edit_area_{idx}", st.session_state.multi_editing_text or turn.content
    )
    st.markdown(
        f"<div class='chat-row {align}'><div class='chat-bubble {role_class} editing' data-role='{role_class}'>",
        unsafe_allow_html=True,
    )
    edit_value = st.text_area(
        "Edit message",
        value=current_text,
        key=f"multi_edit_area_{idx}",
        height=_editor_height(current_text),
        label_visibility="collapsed",
    )
    action_cols = st.columns([1, 1, 6])
    with action_cols[0]:
        save_clicked = st.button(
            "Save",
            key=f"multi-save-edit-{idx}",
            use_container_width=True,
        )
    with action_cols[1]:
        cancel_clicked = st.button(
            "Cancel",
            key=f"multi-cancel-edit-{idx}",
            use_container_width=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)
    return edit_value, save_clicked, cancel_clicked


def _delete_message(index: int) -> bool:
    dialogue = list(st.session_state.dialogue)
    if index < 0 or index >= len(dialogue):
        return False

    deleted_turn = dialogue[index]
    # Drop the deleted message and anything that followed it to mirror edit behaviour
    st.session_state.dialogue = dialogue[:index]
    _clear_edit_state()
    st.session_state.memory_manager = _rebuild_memory(st.session_state.dialogue)
    st.session_state.last_judged = []
    st.session_state.token_usage = {}
    st.session_state.pending_generation = None
    st.session_state.is_thinking = False
    force_title_reset = deleted_turn.role == "user" and index == 0
    _reset_title_seed(force=force_title_reset)
    if deleted_turn.role == "assistant":
        st.session_state.edit_notice = "Assistant reply deleted. Regenerate to continue."
    else:
        st.session_state.edit_notice = "Message deleted. Provide a new prompt to continue."
    return True


def _apply_edit(new_text: str) -> bool:
    index = st.session_state.editing_index
    if index is None:
        return False

    stripped = new_text.strip()
    if not stripped:
        return False

    dialogue = list(st.session_state.dialogue)
    turn = dialogue[index]
    dialogue[index] = Turn(role=turn.role, content=stripped, meta=turn.meta)
    st.session_state.dialogue = dialogue[: index + 1]
    force_title_reset = turn.role == "user" and index == 0
    st.session_state.memory_manager = _rebuild_memory(st.session_state.dialogue)
    st.session_state.last_judged = []
    st.session_state.token_usage = {}
    _reset_title_seed(force=force_title_reset)
    st.session_state.edit_notice = (
        "Message updated. Regenerate the assistant reply to continue."
    )
    _clear_edit_state()
    return True


def _delete_multi_message(index: int) -> bool:
    dialogue = list(st.session_state.multi_dialogue)
    if index < 0 or index >= len(dialogue):
        return False

    deleted_turn = dialogue[index]
    st.session_state.multi_dialogue = dialogue[:index]
    _clear_multi_edit_state()
    st.session_state.multi_memory_manager = _rebuild_memory(st.session_state.multi_dialogue)
    st.session_state.multi_candidates = []
    st.session_state.multi_context_pack = {}
    st.session_state.multi_token_usage = {}
    st.session_state.multi_pending_generation = None
    st.session_state.multi_is_thinking = False
    if deleted_turn.role == "assistant":
        st.session_state.multi_edit_notice = (
            "Assistant reply deleted. Regenerate candidates to continue."
        )
    else:
        st.session_state.multi_edit_notice = (
            "Message deleted. Provide a new prompt to continue."
        )
    return True


def _apply_multi_edit(new_text: str) -> bool:
    index = st.session_state.multi_editing_index
    if index is None:
        return False

    stripped = new_text.strip()
    if not stripped:
        return False

    dialogue = list(st.session_state.multi_dialogue)
    turn = dialogue[index]
    dialogue[index] = Turn(role=turn.role, content=stripped, meta=turn.meta)
    st.session_state.multi_dialogue = dialogue[: index + 1]
    st.session_state.multi_memory_manager = _rebuild_memory(st.session_state.multi_dialogue)
    st.session_state.multi_candidates = []
    st.session_state.multi_context_pack = {}
    st.session_state.multi_token_usage = {}
    st.session_state.multi_pending_generation = None
    st.session_state.multi_is_thinking = False
    if turn.role == "assistant":
        st.session_state.multi_edit_notice = "Assistant reply updated."
    else:
        st.session_state.multi_edit_notice = "Message updated. Regenerate candidates to continue."
    _clear_multi_edit_state()
    return True


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


async def run_candidate_generation(
    settings: Dict[str, Any],
    history: Sequence[Turn],
    memory_manager: MemoryManager,
) -> Tuple[Dict[str, Any], Sequence[Candidate]]:
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
    return context_pack, candidates


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
    st.session_state.edit_notice = ""


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


def _execute_multi_generation(settings: Dict[str, Any], *, new_user: bool) -> None:
    _clear_multi_edit_state()
    st.session_state.multi_edit_notice = ""
    history_snapshot = [Turn(**turn.dict()) for turn in st.session_state.multi_dialogue]
    memory_manager: MemoryManager = st.session_state.multi_memory_manager
    try:
        context_pack, candidates = asyncio.run(
            run_candidate_generation(settings, history_snapshot, memory_manager)
        )
    except Exception as exc:  # pragma: no cover - network error path
        st.error(f"Generation failed: {exc}")
        if new_user and st.session_state.multi_dialogue and st.session_state.multi_dialogue[-1].role == "user":
            st.session_state.multi_dialogue.pop()
        st.session_state.multi_candidates = []
        st.session_state.multi_context_pack = {}
        st.session_state.multi_token_usage = {}
        st.session_state.multi_edit_notice = "Generation failed. Adjust the message and try again."
        return

    st.session_state.multi_candidates = list(candidates)
    st.session_state.multi_context_pack = context_pack
    st.session_state.multi_token_usage = {}
    st.session_state.multi_edit_notice = ""


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


def _apply_multi_selection(index: int) -> bool:
    candidates: Sequence[Candidate] = st.session_state.multi_candidates
    if not candidates or index < 0 or index >= len(candidates):
        return False

    chosen = candidates[index]
    st.session_state.multi_dialogue.append(
        Turn(
            role="assistant",
            content=chosen.text,
            meta={"style": chosen.meta.get("decoding", {}).get("style")},
        )
    )
    usage_summary = {
        "generator": [cand.meta.get("usage") for cand in candidates if cand.meta.get("usage")],
        "judge": None,
    }
    st.session_state.multi_token_usage = {
        "generator": _format_usage(usage_summary["generator"], chosen.text),
        "judge": "n/a",
    }
    context_pack = st.session_state.get("multi_context_pack", {})
    try:
        log_turn(
            context_pack=context_pack,
            candidates=candidates,
            judged=[],
            chosen=chosen.text,
            usage=usage_summary,
            metadata={},
        )
    except Exception:  # pragma: no cover - logging failures should not break UI
        pass
    st.session_state.multi_candidates = []
    st.session_state.multi_context_pack = {}
    st.session_state.multi_edit_notice = ""
    return True


def handle_send(user_text: str) -> bool:
    message = user_text.strip()
    if not message:
        return False
    _clear_edit_state()
    st.session_state.edit_notice = ""
    st.session_state.dialogue.append(Turn(role="user", content=message))
    if not st.session_state.title_generated and not st.session_state.pending_title_prompt:
        st.session_state.pending_title_prompt = message
    return True


def handle_regenerate() -> bool:
    if not st.session_state.dialogue or st.session_state.dialogue[-1].role != "assistant":
        return False
    _clear_edit_state()
    st.session_state.edit_notice = ""
    st.session_state.dialogue.pop()
    return True


def handle_multi_send(user_text: str) -> bool:
    message = user_text.strip()
    if not message:
        return False
    _clear_multi_edit_state()
    st.session_state.multi_edit_notice = ""
    st.session_state.multi_dialogue.append(Turn(role="user", content=message))
    return True


def handle_multi_regenerate() -> bool:
    if not st.session_state.multi_dialogue:
        return False
    if st.session_state.multi_dialogue[-1].role != "user":
        return False
    _clear_multi_edit_state()
    st.session_state.multi_edit_notice = ""
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


def render_judge_chat(settings: Dict[str, Any]) -> None:
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
        chat_placeholder.empty()
        with chat_placeholder.container():
            inject_css()
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for idx, turn in enumerate(st.session_state.dialogue):
                with st.container():
                    if turn.role == "user":
                        message_col, actions_col = st.columns([24, 2])
                    else:
                        actions_col, message_col = st.columns([2, 24])

                    save_clicked = False
                    cancel_clicked = False
                    with message_col:
                        if st.session_state.editing_index == idx:
                            edit_value, save_clicked, cancel_clicked = _render_inline_editor(
                                turn, idx
                            )
                            st.session_state.editing_text = edit_value
                        else:
                            render_message(turn, idx)

                    with actions_col:
                        action_stack = st.container()
                        action_stack.markdown(
                            "<div class='chat-action-stack'>", unsafe_allow_html=True
                        )
                        edit_clicked = action_stack.button(
                            "✏️",
                            key=f"edit-btn-{idx}",
                            help="Edit message",
                        )
                        delete_clicked = action_stack.button(
                            "🗑️",
                            key=f"delete-btn-{idx}",
                            help="Delete message",
                        )
                        action_stack.markdown("</div>", unsafe_allow_html=True)

                    if edit_clicked:
                        _begin_edit(idx)
                        _force_rerun()
                    if delete_clicked:
                        if not _delete_message(idx):
                            st.warning("Unable to delete message.")

                    if st.session_state.editing_index == idx:
                        if save_clicked:
                            if _apply_edit(st.session_state.editing_text):
                                _force_rerun()
                            else:
                                st.warning("Edited message cannot be empty.")
                        if cancel_clicked:
                            _clear_edit_state()
                            st.session_state.edit_notice = ""
                            _force_rerun()
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

    draw_chat(thinking=bool(st.session_state.get("is_thinking")))

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
        if handle_send(user_text):
            st.session_state.last_judged = []
            st.session_state.token_usage = {}
            st.session_state.pending_generation = {
                "settings": dict(settings),
                "new_user": True,
                "label": "Generating assistant reply…",
            }
            st.session_state.is_thinking = True
            _trigger_rerun()
            return
        st.warning("Please enter a message before sending.")
    elif regenerate_clicked:
        if handle_regenerate():
            st.session_state.last_judged = []
            st.session_state.token_usage = {}
            st.session_state.pending_generation = {
                "settings": dict(settings),
                "new_user": False,
                "label": "Regenerating assistant reply…",
            }
            st.session_state.is_thinking = True
            _trigger_rerun()
            return
        st.info("Nothing to regenerate yet.")
    elif stop_clicked:
        st.info("No generation in progress to stop.")

    pending = st.session_state.get("pending_generation")
    if pending:
        label = str(pending.get("label", "Generating assistant reply…"))
        pending_settings = dict(pending.get("settings", {}))
        new_user = bool(pending.get("new_user", True))
        with st.spinner(label):
            _execute_generation(pending_settings, new_user=new_user)
        st.session_state.pending_generation = None
        st.session_state.is_thinking = False
        _trigger_rerun()
        return

    if st.session_state.show_candidates and st.session_state.last_judged:
        st.markdown("### Candidate Rankings")
        render_candidates(st.session_state.last_judged)

    if st.session_state.edit_notice:
        st.info(st.session_state.edit_notice)


def render_multi_choice_chat(settings: Dict[str, Any]) -> None:
    st.markdown("<div class='chat-header'><h2>Multi-choice Chat</h2></div>", unsafe_allow_html=True)
    usage = st.session_state.multi_token_usage
    if usage:
        st.markdown(
            f"<div class='token-counter'>Usage · Generator: {usage.get('generator', 'n/a')}</div>",
            unsafe_allow_html=True,
        )

    chat_placeholder = st.empty()

    def draw_multi_chat(thinking: bool = False) -> None:
        chat_placeholder.empty()
        with chat_placeholder.container():
            inject_css()
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for idx, turn in enumerate(st.session_state.multi_dialogue):
                with st.container():
                    if turn.role == "user":
                        message_col, actions_col = st.columns([24, 2])
                    else:
                        actions_col, message_col = st.columns([2, 24])

                    save_clicked = False
                    cancel_clicked = False
                    with message_col:
                        if st.session_state.multi_editing_index == idx:
                            edit_value, save_clicked, cancel_clicked = _render_multi_inline_editor(
                                turn, idx
                            )
                            st.session_state.multi_editing_text = edit_value
                        else:
                            render_message(turn, idx)

                    with actions_col:
                        action_stack = st.container()
                        action_stack.markdown(
                            "<div class='chat-action-stack'>", unsafe_allow_html=True
                        )
                        edit_clicked = action_stack.button(
                            "✏️",
                            key=f"multi-edit-btn-{idx}",
                            help="Edit message",
                        )
                        delete_clicked = action_stack.button(
                            "🗑️",
                            key=f"multi-delete-btn-{idx}",
                            help="Delete message",
                        )
                        action_stack.markdown("</div>", unsafe_allow_html=True)

                    if edit_clicked:
                        _begin_multi_edit(idx)
                        _force_rerun()
                    if delete_clicked:
                        if not _delete_multi_message(idx):
                            st.warning("Unable to delete message.")

                    if st.session_state.multi_editing_index == idx:
                        if save_clicked:
                            if _apply_multi_edit(st.session_state.multi_editing_text):
                                _force_rerun()
                            else:
                                st.warning("Edited message cannot be empty.")
                        if cancel_clicked:
                            _clear_multi_edit_state()
                            st.session_state.multi_edit_notice = ""
                            _force_rerun()
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

    draw_multi_chat(thinking=bool(st.session_state.get("multi_is_thinking")))

    with st.form("multi-chat-input", clear_on_submit=True):
        user_text = st.text_area("Message", height=120, key="multi_chat_input")
        cols = st.columns([1, 1])
        send_clicked = cols[0].form_submit_button("Send", type="primary")
        regenerate_clicked = cols[1].form_submit_button("Regenerate candidates")

    if send_clicked:
        if st.session_state.multi_candidates:
            st.warning("Select or regenerate the existing candidates before sending a new message.")
        elif st.session_state.multi_dialogue and st.session_state.multi_dialogue[-1].role == "user":
            st.warning("Awaiting assistant response. Choose a candidate or regenerate.")
        elif handle_multi_send(user_text):
            st.session_state.multi_pending_generation = {
                "settings": dict(settings),
                "new_user": True,
                "label": "Generating candidates…",
            }
            st.session_state.multi_is_thinking = True
            _trigger_rerun()
            return
        else:
            st.warning("Please enter a message before sending.")
    elif regenerate_clicked:
        if not handle_multi_regenerate():
            st.info("Nothing to regenerate yet.")
        else:
            st.session_state.multi_pending_generation = {
                "settings": dict(settings),
                "new_user": False,
                "label": "Regenerating candidates…",
            }
            st.session_state.multi_is_thinking = True
            _trigger_rerun()
            return

    pending = st.session_state.get("multi_pending_generation")
    if pending:
        label = str(pending.get("label", "Generating candidates…"))
        pending_settings = dict(pending.get("settings", {}))
        new_user = bool(pending.get("new_user", True))
        with st.spinner(label):
            _execute_multi_generation(pending_settings, new_user=new_user)
        st.session_state.multi_pending_generation = None
        st.session_state.multi_is_thinking = False
        _trigger_rerun()
        return

    if st.session_state.multi_candidates:
        st.markdown("### Choose a response")
        for idx, candidate in enumerate(st.session_state.multi_candidates):
            button_label = f"Option {idx + 1}\n{candidate.text}"
            if st.button(
                button_label,
                key=f"multi-select-{idx}",
                use_container_width=True,
            ):
                if _apply_multi_selection(idx):
                    st.session_state.multi_is_thinking = False
                    _trigger_rerun()
                    return

    if st.session_state.multi_edit_notice:
        st.info(st.session_state.multi_edit_notice)

def main() -> None:
    init_state()
    settings = st.session_state.settings
    st.session_state.settings = render_sidebar(settings)
    judge_tab, multi_tab = st.tabs(["Judge chat", "Multi-choice chat"])
    with judge_tab:
        render_judge_chat(st.session_state.settings)
    with multi_tab:
        render_multi_choice_chat(st.session_state.settings)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
