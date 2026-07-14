# Multi-Agent Paper Reader

**English** | [简体中文](./README.zh-CN.md)

Multi-Agent Paper Reader is an evidence-grounded research paper reading assistant. Upload a PDF, parse its sections, build a traceable evidence index from text, extracted tables, and vision-model figure summaries, run specialist agents for method analysis, experiment analysis, and critique, then synthesize a polished structured reading note.

![Paper Reader workspace](./docs/assets/paper-reader-workspace.png)

## Web App

The repository includes a full-stack web app:

- Backend: `app.py` with FastAPI
- Frontend: `frontend-prototype/` with React + Vite
- API: `POST /api/analyze` accepts a PDF upload and returns parsed paper metadata plus all agent outputs
- Streaming API: `POST /api/analyze/stream` returns newline-delimited JSON events for parsing, evidence indexing, token-level model output, agent completion, and final synthesis
- Follow-up API: `POST /api/chat/stream` combines recent turns, a compact long-term memory index, query-relevant topic memories, recalled older messages, and full-text paper evidence
- Conversation API: `GET/POST /api/history/{id}/conversations` plus `GET/PATCH/DELETE /api/chat/conversations/{id}` support multiple persistent chats per paper
- Comparison API: `POST /api/comparisons/stream` compares 2-4 saved papers with prefixed evidence, while `/api/comparisons/*` persists comparison workspaces and cross-paper conversations
- History API: `GET /api/history`, `GET /api/history/{id}`, and `DELETE /api/history/{id}` persist and restore completed analyses
- Settings API: `GET /api/settings` returns provider, protocol, and active-route metadata without exposing credentials; credential saves must pass a minimal real text-model request over the selected protocol before local persistence
- Section titles: the chapter list preserves each title in its original parsed language, compacting only abnormal whitespace and falling back to a numbered placeholder for damaged text
- Static hosting: the FastAPI server serves the built React app from `frontend-prototype/dist`

Run it locally:

```bash
# Python backend dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Frontend dependencies and production build
cd frontend-prototype
npm install
npm run build
cd ..

# Start the full-stack app
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

For first-time setup, open **Settings** in the top-right navigation. Built-in routes use concise provider names: GLM, DeepSeek, OpenAI, Qwen, Doubao, Anthropic, and Kimi. Catalog entries use the real IDs documented by each provider, including the current Claude Fable 5 / Sonnet 5 / Opus 4.8, Kimi K2.6, Doubao Seed 2.1 Pro / Turbo, GPT-5.6 Sol, Qwen3.7, and DeepSeek V4 lines. GLM-5.2 exposes standard, deep, and fast modes; deep mode sends `reasoning_effort=max`. Qwen3.7/3.6 hybrid-thinking models switch with `enable_thinking`; deep mode leaves `thinking_budget` unset so the provider's documented model maximum applies. Kimi K2.6 and DeepSeek V4 send the real `thinking.type` field. Vision always follows the text provider and is enabled only when that service has an explicit vision model.

Settings also includes a custom relay route. Users must explicitly select `OpenAI-compatible` or `Anthropic-compatible`, then provide a Base URL, text model ID, and optional vision model ID. Leaving the vision model blank explicitly disables rendered-image understanding. A minimal real text request must succeed before the API key and relay metadata are persisted to the local `.env`; saved keys are never returned to the browser. Relay traffic is sent to the configured third party, so users remain responsible for its privacy, billing, and reliability.

The default route remains Zhipu `glm-5.2` for text and its automatic `glm-5v-turbo` vision pairing. You can also copy `.env.example` to `.env` and configure `TEXT_PROVIDER`, `MODEL_NAME`, and provider-specific keys manually. `VISION_PROVIDER` is retained for backward compatibility but is normalized to `TEXT_PROVIDER`; built-in providers use their recommended vision model, while a custom relay uses the explicit user-supplied vision model ID. Agent generation uses `LLM_TEMPERATURE`; grounded follow-up chat has its own lower `CHAT_TEMPERATURE` (default `0.25`). `CHAT_INPUT_TOKEN_BUDGET` sets the conservative dynamic input budget used to balance evidence, recent turns, and long-term memory (default `48000`).

For figure/chart understanding, set `ENABLE_VISION_SUMMARY=true` and select a text provider that offers a hosted vision model. The backend renders PDF visual regions to PNG, fans out one concurrent vision request per selected figure/chart by default, asks the paired vision model for concise Chinese visual summaries, and indexes them as `F` evidence. If the provider returns rate-limit errors, failed figures are automatically retried with the smaller `VISION_RETRY_WORKERS` pool.

## CLI Quick Start

```bash
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set the API key for the selected provider

python main.py examples/your_paper.pdf --pretty
```

## Validation

Every push to `main` runs backend tests and a Vite production build on macOS, Windows, and Ubuntu through `.github/workflows/ci.yml`. For credentialed release checks, `tools/provider_smoke.py` performs both a minimal real request and an evidence-citing paper follow-up through the application chat path while printing only non-secret trace fields:

```bash
python tools/provider_smoke.py openai deepseek doubao
```

The same check is available as the manually triggered `Live provider smoke tests` workflow when `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, and `ARK_API_KEY` repository secrets are configured.

## Architecture

```text
PDF
-> core.pdf_parser.parse_pdf
-> core.evidence.build_evidence_index
-> MethodAgent + ExperimentAgent + CriticAgent read relevant text/table/figure evidence snippets
-> SummaryAgent synthesizes structured notes with carried-forward evidence
-> structured reading note
```

The live stream emits these event types:

- `paper`
- `evidence_index`
- `vision_started`
- `vision_complete`
- `vision_error`
- `agent_started`
- `agent_token`
- `agent_complete`
- `complete`
- `error`

`agent_token` is the raw model generation stream. The frontend shows it as a live preview, then renders the parsed Pydantic output once the JSON object is complete.

## Multi-Paper Comparison

Use the `Reading Workspace` menu to switch to `Comparison Workspace`, then select two to four previously analyzed papers. ComparisonAgent reuses each paper's structured Method, Experiment, Critic, and Summary outputs, but verifies paper-specific claims against balanced slices of the saved original evidence. Evidence IDs are namespaced as `P1:E003`, `P2:T001`, or `P3:F002`, so overlapping IDs from different PDFs cannot be confused.

The result includes a horizontally scrollable comparison matrix, direct/conditional/not-comparable labels, dataset and metric mismatch warnings, clickable evidence previews, research gaps, conditional recommendations, and deterministic evidence-coverage indicators. Completed comparisons, selected paper relationships, and cross-paper conversations are stored in the same SQLite database and survive browser refreshes or backend restarts. Cross-paper chat retrieves a balanced evidence set from every selected paper instead of sending all PDFs to the model at once.

## Explainable Assessment

Every completed API response includes an `assessment` object with two separate results:

- **Novelty (1-5):** the Critic Agent scores problem originality (15%), method originality (40%), difference from prior work (30%), and generality (15%). The backend calculates the weighted total and keeps each reason and supporting evidence ID.
- **Analysis reliability (0-100):** the backend deterministically scores PDF parsing (20), key-section coverage (35), valid evidence citations (30), and structured-output integrity (15).

Reliability is not the model's self-reported confidence. Missing related-work coverage, fewer than three valid citations, insufficient parsed content, incomplete novelty dimensions, or Demo mode apply explicit score caps. The response exposes the component scores, raw score, cap, final score, and warnings so the result can be audited.

## Follow-up Chat

Open the paper chat directly from the AI button at the bottom-right of the results panel, or select text and choose **在侧边聊天中提问** to include that excerpt. Each paper can have multiple independent conversations, selectable from the chat header. Every original user and assistant message is stored in SQLite and restored after browser refreshes or backend restarts.

The composer can route each question through any provider whose key has already been verified and saved, with a model-specific response mode when available. This request-scoped selection does not rewrite the global Settings route. The selector and answer footer show the model name once, without verification text, expandable provider endpoints, upstream model metadata, or request IDs. Every assistant answer has a copy action. Paper evidence IDs and page labels still constrain retrieval and generation internally, but are removed from the user-visible answer.

The first question receives an immediate concise local title, then a background GLM request refines it into a short topic summary without overwriting a manual rename. Chat Markdown supports GFM tables and KaTeX-rendered inline or display mathematics. During streaming, new tokens follow the viewport only while the reader is already near the bottom; scrolling upward pauses auto-follow and exposes a one-click return-to-latest control.

The long-conversation design follows the public Claude Code memory pattern: a concise memory index is loaded on every turn, while detailed topic memories are recalled only when relevant. The latest six complete rounds remain verbatim. Once another six older rounds become eligible, a background GLM task compacts them into the memory index and topic records without deleting the original messages or delaying the current answer. Each question can additionally recall relevant older original messages. A dynamic token budget prioritizes the current question, selected excerpt, original paper evidence, recent turns, analysis context, memory, and external sources instead of relying on a fixed message-count cutoff.

Each completed Live analysis returns an opaque `analysis_id`; the backend keeps that analysis's complete `E`/`T`/`F` evidence snippets in a bounded four-hour in-memory cache. For every question it combines Chinese/English query terms, conversational context, Agent-cited evidence IDs, and section intent to retrieve the most relevant original snippets. The answer prompt treats original paper evidence as authoritative and explicitly distinguishes paper facts, background knowledge, memory, and inference, while keeping internal evidence identifiers out of the final display.

Questions that explicitly ask for recent work, related papers, or comparisons with other papers can also use Semantic Scholar title/abstract metadata. This lookup is optional and fails closed; `SEMANTIC_SCHOLAR_API_KEY` can be configured for a dedicated API quota. Sample and Demo results use a deterministic reply so the complete interaction can be tested without another model call. Live in-memory chat sessions are recreated from persisted full evidence whenever a saved paper is reopened.

## Paper History

Every completed upload is saved locally in `.paper-reader/` by default. The SQLite database contains paper metadata, structured Agent outputs, assessment results, complete evidence snippets, single-paper and comparison workspaces, chat conversations, immutable original messages, memory indexes, and topic memories; the original PDF is retained in `.paper-reader/papers/`. Uploading the same PDF again updates its existing history record instead of creating a duplicate. `Recent Papers` and the header History menu read this database, so a saved analysis and all of its conversations can be reopened after a browser refresh or backend restart without uploading the PDF again. Reopening a Live result recreates its grounded evidence session from the saved evidence.

Set `PAPER_READER_DATA_DIR` to move all history storage, or set `PAPER_HISTORY_DB` to choose a specific SQLite path. Deleting an item from the History menu removes both its database record and retained PDF.

See [CLAUDE.md](./CLAUDE.md) for the original architecture notes.

## Tech Stack

- LangGraph for agent orchestration
- PyMuPDF for PDF parsing, outline-based section detection, table extraction, and rendered figure regions
- Pydantic v2 for output schema validation
- Evidence snippets for page/section-grounded claims (`E` text, `T` table, `F` vision figure summary)
- FastAPI for the backend API and static hosting
- React + Vite for the frontend workbench
