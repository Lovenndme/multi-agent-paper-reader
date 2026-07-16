# Multi-Agent Paper Reader

**English** | [简体中文](./README.zh-CN.md)

Multi-Agent Paper Reader is an evidence-grounded research paper reading assistant. Upload a PDF, parse its sections, build a traceable evidence index from text, extracted tables, and vision-model figure summaries, run specialist agents for method analysis, experiment analysis, and critique, then synthesize a polished structured reading note.

![Paper Reader workspace](./docs/assets/paper-reader-workspace.png)

## Web App

The repository includes a full-stack web app:

- Backend: `app.py` with FastAPI
- Frontend: `frontend-prototype/` with React + Vite
- API: `POST /api/analyze` accepts a PDF upload and returns parsed paper metadata plus all agent outputs
- Preview API: `POST /api/papers/preview` immediately returns the paper title, page count, section count, and original section list without model calls or history writes
- Streaming API: `POST /api/analyze/stream` returns newline-delimited JSON events for parsing, evidence indexing, token-level model output, agent completion, and final synthesis
- Follow-up API: `POST /api/chat/stream` combines recent turns, a compact long-term memory index, query-relevant topic memories, recalled older messages, and full-text paper evidence
- Conversation API: `GET/POST /api/history/{id}/conversations` plus `GET/PATCH/DELETE /api/chat/conversations/{id}` support multiple persistent chats per paper
- Comparison API: `POST /api/comparisons/stream` compares 2-4 saved papers with prefixed evidence, while `/api/comparisons/*` persists comparison workspaces and cross-paper conversations
- History API: `GET /api/history`, `GET /api/history/{id}`, and `DELETE /api/history/{id}` persist and restore completed analyses
- Settings API: `GET /api/settings` returns provider, protocol, and active-route metadata without exposing credentials; `/api/settings/codex/*` reports the local Codex login/model state and starts a localhost-only ChatGPT login flow; API-key saves must pass a minimal real text-model request before local persistence
- Section titles: the chapter list preserves each title in its original parsed language, compacting only abnormal whitespace and falling back to a numbered placeholder for damaged text
- Static hosting: the FastAPI server serves the built React app from `frontend-prototype/dist`

Python 3.10 or later is required. Git/source checkouts also need Node.js 18 or
later with npm so the updater can rebuild the frontend. The formal Release ZIP
already includes a verified frontend build and does not require Node.js unless
that build is missing or mismatched. A Git clone is recommended for developers
because it can be updated safely in place.

First-time setup on Windows PowerShell:

```powershell
git clone https://github.com/Lovenndme/multi-agent-paper-reader.git
cd multi-agent-paper-reader
powershell -ExecutionPolicy Bypass -File .\scripts\update.ps1
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

First-time setup on macOS or Linux:

```bash
git clone https://github.com/Lovenndme/multi-agent-paper-reader.git
cd multi-agent-paper-reader
bash ./scripts/update.sh
./.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

### Updating an existing installation

Stop the running service and run the updater from the project root. On Windows
PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\update.ps1
```

On macOS or Linux:

```bash
bash ./scripts/update.sh
```

In a Git clone, the updater runs `git pull --ff-only` only when tracked files
and the index are clean. It does not delete untracked files or overwrite the
local `.env` and `.paper-reader/` data. It then installs the pinned Python
requirements, runs `npm ci`, rebuilds the production frontend, and verifies
`frontend-prototype/dist/build-meta.json` against the backend
`PROJECT_VERSION`. This version check is a hard gate: missing, invalid, or
mismatched metadata makes the script exit non-zero, so the update has not
succeeded. Restart the service with the command printed by the script; an old
process cannot serve the new code.

If the installation came from an archive, it has no Git metadata and the script
deliberately cannot download a newer source tree. Download the latest
`Paper-Reader-<version>.zip` and its matching `.zip.sha256` from GitHub Releases,
verify the checksum, and extract the archive. Copy the existing `.env` and
`.paper-reader/` into the new project directory as a migration (keep the old
directory as a backup), then run the updater there. The release archive already
contains the matching production frontend, so the updater verifies and reuses
it without requiring Node.js. Do not copy an old `frontend-prototype/dist`
directory into the new release.

For first-time setup, open **Settings** in the top-right navigation. Built-in routes use concise provider names: GLM, DeepSeek, OpenAI, Qwen, Doubao, Anthropic, Kimi, and Codex Subscription. API-provider catalog entries use the real IDs documented by each provider, including the current Claude Fable 5 / Sonnet 5 / Opus 4.8, Kimi K2.6, Doubao Seed 2.1 Pro / Turbo, GPT-5.6 Sol, Qwen3.7, and DeepSeek V4 lines. GLM-5.2 exposes standard, deep, and fast modes; deep mode sends `reasoning_effort=max`. Qwen3.7/3.6 hybrid-thinking models switch with `enable_thinking`; deep mode leaves `thinking_budget` unset so the provider's documented model maximum applies. Kimi K2.6 and DeepSeek V4 send the real `thinking.type` field. Vision always follows the text provider and is enabled only when that service has an explicit vision model.

### Codex subscription route (local single-user only)

The optional Codex route uses an immutable snapshot of the official `openai-codex` Python SDK, pinned together with runtime `0.144.4` in `requirements.txt`. This source revision can decode the current GPT-5.6 `max` and `ultra` catalog values; the runtime, source revision, archive hash, and installed binary version are checked before the route is enabled. Python 3.10 or later is required, but a separate global Codex CLI installation is not. See the official [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk), [model](https://learn.chatgpt.com/docs/models), and [authentication](https://learn.chatgpt.com/docs/auth) documentation, plus the [pinned SDK source revision](https://github.com/openai/codex/commit/3f74f00295dcb1346340686bb09c5bfd4f0237c4).

Open **Settings → Codex Subscription** and use either the official browser login or device-code flow. An existing local Codex/ChatGPT session is reused; otherwise the official runtime completes login and stores authentication in its normal local cache. The web app never receives, reads, persists, or returns a Codex token. Disconnecting explicitly logs out that shared local session, so the UI warns that other local Codex clients may need to sign in again.

Only the account's live `model/list` response is authoritative. The route exposes `gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.6-luna` when the signed-in account actually returns them; it never silently falls back to stale or static subscription models. Settings always presents six reasoning positions—`low`, `medium`, `high`, `xhigh`, `max`, and `ultra`—and disables any position omitted by the selected model. In the currently verified catalog, Sol and Terra support all six, while Luna supports the first five and displays Ultra as unavailable.

Codex can power specialist agents, structured summaries, single-paper and comparison chat, conversation titles, image/chart summaries, and JSON-Schema memory updates. Every call uses an isolated ephemeral thread, denied approvals, a read-only sandbox, and disabled Codex history persistence. Native Web Search, planning, tool discovery, image viewing, and image generation remain available when supported by the selected model and pinned runtime; external web sources are displayed separately from paper evidence, and image generation is reserved for explicit user requests. Turns bound to one paper additionally receive ten host-provided read-only tools: paper overview and asset discovery, evidence search, section/page text lookup, explicit single-page images, figure/table metadata, parser-verified visual crops, paper-scoped memory recall, and a bounded calculator. The integration disables inherited MCP servers, apps, plugins, hooks, memories, skill injection, installation suggestions, interactive prompts, Shell, unified exec, and arbitrary filesystem writes, then verifies the effective MCP catalog is isolated before starting the SDK. Generated images are the only controlled write exception and are stored by the official runtime under `CODEX_HOME/generated_images`. Ultra may create at most two child agents at depth one; every other effort uses a one-thread capacity, so model-advertised collaboration tools cannot create a child. See the official [approval and sandbox guidance](https://learn.chatgpt.com/docs/agent-approvals-security) and [MCP documentation](https://learn.chatgpt.com/docs/extend/mcp).

This route is intentionally limited to a trusted single-user application reached through a loopback address. It is not a hosted multi-user authentication scheme and must not let remote visitors consume the machine owner's ChatGPT subscription. Login/logout endpoints require a loopback client, a loopback Host, and same-origin browser requests to resist cross-site and DNS-rebinding attacks. Public or shared deployments must use normal provider API credentials with their own user authorization, quotas, and billing.

The complete setup, routing, tool matrix, and upgrade checklist are documented in [docs/codex-subscription.md](./docs/codex-subscription.md).

Settings also includes a custom relay route. Users must explicitly select `OpenAI-compatible` or `Anthropic-compatible`, then provide a Base URL, text model ID, and optional vision model ID. Leaving the vision model blank explicitly disables rendered-image understanding. A minimal real text request must succeed before the API key and relay metadata are persisted to the local `.env`; saved keys are never returned to the browser. Relay traffic is sent to the configured third party, so users remain responsible for its privacy, billing, and reliability.

The default route remains Zhipu `glm-5.2` for text and its automatic `glm-5v-turbo` vision pairing. You can also copy `.env.example` to `.env` and configure `TEXT_PROVIDER`, `MODEL_NAME`, and provider-specific keys manually. `VISION_PROVIDER` is retained for backward compatibility but is normalized to `TEXT_PROVIDER`; built-in providers use their recommended vision model, while a custom relay uses the explicit user-supplied vision model ID. Agent generation uses `LLM_TEMPERATURE`; grounded follow-up chat has its own lower `CHAT_TEMPERATURE` (default `0.25`). `CHAT_INPUT_TOKEN_BUDGET` sets the conservative dynamic input budget used to balance evidence, recent turns, and long-term memory (default `48000`).

For figure/chart understanding, set `ENABLE_VISION_SUMMARY=true` and select a text provider that offers a hosted vision model. PyMuPDF4LLM Layout classifies body text, formulas, tables, captions, and picture regions; pictures cover both embedded raster images and PDF vector graphics. Captions are matched to same-page pictures geometrically instead of by unrelated list positions. The vision model receives only a verified bounding-box crop rendered through the bounded 120-144 DPI preview path; a missing region is skipped and never silently rendered as a full page. Native/export rendering is independent: raster figures preserve the highest overlapping source density, vector figures and tables use a 600 DPI quality floor, and an over-limit render is refused instead of silently downsampled. Rate-limited figures are retried with the smaller `VISION_RETRY_WORKERS` pool.

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
-> PyMuPDF4LLM Layout classifies body, formulas, tables, captions, raster and vector figures
-> core.pdf_parser.parse_pdf builds original-language sections and verified visual regions
-> core.evidence.build_evidence_index + local multilingual embedding ranking
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

The composer can route each question through any provider whose key has already been verified and saved, or through a connected local Codex subscription, with a model-specific response mode when available. This request-scoped selection does not rewrite the global Settings route. The selector and answer footer show the model name once, without verification text, expandable provider endpoints, upstream model metadata, or request IDs. Every assistant answer has a copy action. Paper evidence IDs and page labels still constrain retrieval and generation internally, but are removed from the user-visible answer.

The first question receives an immediate concise local title, then a background GLM request refines it into a short topic summary without overwriting a manual rename. Chat Markdown supports GFM tables and KaTeX-rendered inline or display mathematics. During streaming, new tokens follow the viewport only while the reader is already near the bottom; scrolling upward pauses auto-follow and exposes a one-click return-to-latest control.

The memory layer now uses LangMem 0.0.30 as its sole long-term-memory engine. After each completed answer, a background LangMem manager extracts, updates, or deletes structured `user`, `feedback`, `project`, and `reference` memories. Memories are scoped per paper, persisted in the existing SQLite database through a LangGraph `BaseStore` adapter, indexed locally with multilingual MiniLM embeddings, and retrieved by cosine similarity with a no-match threshold. A normal paper question therefore adds no model-side memory-selection call before the answer. Explicit ignore-memory requests inject no long-term context. Raw SQLite messages remain available for conversation history but are never reintroduced as long-term recall, so deleted memories cannot silently resurface from old messages. Legacy SQLite and file-based memories are imported once for compatibility.

See [`docs/claude-memory-port.md`](docs/claude-memory-port.md) for the source-mechanism compliance matrix and runtime bindings.

Each completed Live analysis returns an opaque `analysis_id`; the backend keeps that analysis's complete `E`/`T`/`F` evidence snippets in a bounded four-hour in-memory cache. Agent evidence selection, paper chat, comparison, old-message recall, and topic-memory recall use a local multilingual embedding as the primary ranker. First use downloads about 240 MB of `paraphrase-multilingual-MiniLM-L12-v2` into `.paper-reader/models/`; paper text stays local and no provider key is required. Lexical retrieval is retained only as an explicit fallback. Set `PAPER_READER_MODEL_DIR` to move the cache or `PAPER_READER_DISABLE_EMBEDDINGS=true` to disable embeddings. The answer prompt treats original paper evidence as authoritative and distinguishes paper facts, background knowledge, memory, and inference.

Questions that explicitly ask for recent work, related papers, or comparisons with other papers can also use Semantic Scholar title/abstract metadata. This lookup is optional and fails closed; `SEMANTIC_SCHOLAR_API_KEY` can be configured for a dedicated API quota. Sample and Demo results use a deterministic reply so the complete interaction can be tested without another model call. Live in-memory chat sessions are recreated from persisted full evidence whenever a saved paper is reopened.

## Paper History

Every completed upload is saved locally in `.paper-reader/` by default. SQLite contains paper metadata, structured Agent outputs, assessment results, complete evidence snippets, single-paper and comparison workspaces, chat conversations, and immutable original messages. Original PDFs live in `.paper-reader/papers/`, while layered file memory lives in `.paper-reader/memory/`. Uploading the same PDF updates its existing history record instead of creating a duplicate. `Recent Papers` and History restore saved analyses and conversations after browser or backend restarts.

Set `PAPER_READER_DATA_DIR` to move all history storage, or set `PAPER_HISTORY_DB` to choose a specific SQLite path. Deleting an item from the History menu removes both its database record and retained PDF.

See [CLAUDE.md](./CLAUDE.md) for the original architecture notes.

## Tech Stack

- LangGraph for agent orchestration
- OpenAI Codex Python SDK for optional local ChatGPT-subscription-backed inference
- PyMuPDF4LLM Layout + PyMuPDF for body/formula/table/caption classification, raster/vector figure detection, and precise rendering
- FastEmbed for local multilingual semantic retrieval; reliability remains a deterministic auditable score
- Pydantic v2 for output schema validation
- Evidence snippets for page/section-grounded claims (`E` text, `T` table, `F` vision figure summary)
- FastAPI for the backend API and static hosting
- React + Vite for the frontend workbench
