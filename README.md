# LLM Judge Chat

```
+------------------+       +---------------------+       +----------------+
| Streamlit Client |  -->  | Candidate Generator |  -->  | Judge & Logger |
+------------------+       +---------------------+       +----------------+
         ^                                                        |
         |                                                        v
         +-------------------- Memory & Context ------------------+
```

LLM Judge Chat is a local-first conversation orchestrator that produces multiple
candidate replies from a configurable OpenAI-compatible model, scores them with
an impartial judge model, selects the best, and presents the result through an
LM Studio-inspired Streamlit interface. Every turn is archived to JSONL for
auditing and downstream dataset preparation.

## Features

* Parallel generation of 3–5 stylistically diverse candidate replies.
* Judge model scoring across in-character fidelity, continuity, emotional
  realism, scene advancement, and coherence with weighted aggregation and
  concise rationales.
* Automatic selection of the top candidate plus optional fallback heuristics.
* Rolling memory heuristics capturing user facts, preferences, and TODOs.
* JSONL logging per turn and DPO pair export utility.
* Streamlit UI mirroring LM Studio’s layout with sidebar controls, chat
  bubbles, regeneration, and score inspection.
* Editable system prompts for both assistant and judge models directly in the
  sidebar, enabling persona tweaks and custom evaluation goals without code.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
```

Edit `.env` with your generator and judge credentials or configure them from
the UI sidebar.

## Configuration

Both generator and judge clients target OpenAI-compatible `/chat/completions`
endpoints. Point the app at local runtimes such as LM Studio, vLLM, Oobabooga
(OpenAI API plugin), or any OpenRouter-compatible service by setting the base
URL and API key fields. The sidebar exposes decoding knobs and connection tests
for each server. Settings can be persisted back to `.env` via the “Save as
defaults” button.

The generator and judge may use different servers or models. Customize their
system prompts in the sidebar to enforce desired personas, values, or judging
criteria. For example:

* Generator: local LM Studio server at `http://localhost:1234/v1` with model
  `mistral-7b-instruct`.
* Judge: remote OpenRouter endpoint using a stronger model such as `gpt-4`.

## Running the UI

```bash
streamlit run src/ui/app.py
```

Keyboard shortcuts: **Enter** to send, **Shift+Enter** for newline. Toggle the
candidate inspection panel to compare ranked responses and rationales.

## Logs & DPO preparation

Each assistant turn appends a structured record to `logs/session_YYYYMMDD.jsonl`
including context, candidates, judge scores, selection, and usage metadata. To
convert logs into Direct Preference Optimization pairs:

```bash
python -m llm_judge_chat.dpo_prep --logs_dir logs --out pairs.jsonl --min_gap 0.5
```

The exporter keeps comparisons where the judge score gap between the chosen and
rejected candidate is at least `min_gap` (default 0.5).

## Performance tips

* Reduce `n_candidates` or `max_tokens` in the sidebar to shorten latency.
* Adjust timeout seconds if using slower local models.
* Disable the candidate panel when not inspecting scores to lighten the UI.

## Known limitations

* Memory heuristics are lightweight string matchers; they may miss nuanced
  user traits.
* The fallback reranker is simplistic—ensure judge endpoints are reliable for
  best quality.
* “Stop” toggles the running flag but cannot interrupt an in-flight HTTP call
  mid-request.
* Token usage estimates rely on API responses when available; otherwise a
  heuristic token counter is used.

## License

MIT License. See `pyproject.toml` for dependency information.
