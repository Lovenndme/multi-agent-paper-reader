# AGENT.md

This file is for Codex or another coding agent working in this repository.

Last reviewed: 2026-07-08

## Project Summary

This repo is a Python project named `multi-agent-paper-reader` with a CLI pipeline and a React/FastAPI web app.

It reads an academic paper PDF, extracts sections, builds an evidence index from text, table, and vision-model figure summaries, runs three specialist agents in parallel over relevant evidence snippets, and synthesizes their outputs into structured paper reading notes.

Current pipeline:

```text
PDF
-> core.pdf_parser.parse_pdf
-> core.evidence.build_evidence_index
-> MethodAgent + ExperimentAgent + CriticAgent
-> SummaryAgent
-> core.schemas.SummaryOutput JSON
```

The web app entrypoint is `app.py`; it exposes API endpoints and serves the built frontend from `frontend-prototype/dist`.

## Current Reality Check

Remote:

- `origin`: `https://github.com/Lovenndme/multi-agent-paper-reader.git`
- current branch at inspection time: `main`
- inspected commit: `8097870 Fix four issues from code review`

Files that exist:

- `main.py`
- `core/schemas.py`
- `core/pdf_parser.py`
- `core/graph.py`
- `agents/method_agent.py`
- `agents/experiment_agent.py`
- `agents/critic_agent.py`
- `agents/summary_agent.py`
- `prompts/method.txt`
- `prompts/experiment.txt`
- `prompts/critic.txt`
- `prompts/summary.txt`
- `utils/llm.py`
- `tests/test_pdf_parser.py`
- `README.md`
- `CLAUDE.md`
- `.env.example`
- `requirements.txt`
- `app.py`
- `frontend-prototype/`

Files or folders mentioned in docs but not currently present:

- `examples/`
- `docs/`

If you add one of those, update docs and tests accordingly.

## Setup Commands

Use a virtual environment if doing real development:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The project needs an `.env` file for LLM calls:

```bash
copy .env.example .env
```

Required environment variable:

- `GLM_API_KEY`

Supported environment variables:

- `GLM_BASE_URL`, default `https://open.bigmodel.cn/api/paas/v4`
- `MODEL_NAME`, default `glm-5.2`
- `LLM_TEMPERATURE`, default `1.0` in `utils/llm.py`

Legacy `OPENAI_API_KEY` and `OPENAI_BASE_URL` variables remain supported for other OpenAI-compatible providers.

## Run Commands

Run the pipeline:

```bash
python main.py path\to\paper.pdf --pretty
```

Save JSON output:

```bash
python main.py path\to\paper.pdf --pretty --output notes.json
```

CLI contract:

- positional argument: `pdf`
- optional `--output` / `-o`
- optional `--pretty`
- stdout: JSON output when `--output` is not provided
- stderr: progress logs and human-readable summary

Run the full-stack web app:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend-prototype
npm install
npm run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

`POST /api/analyze` accepts multipart PDF uploads. With `demo=true`, it runs the parser and returns deterministic mock agent outputs. Without `demo=true`, it runs the real LangGraph pipeline and requires `GLM_API_KEY` (or the legacy `OPENAI_API_KEY`).

`POST /api/analyze/stream` is the preferred frontend path. It returns newline-delimited JSON events:

- `paper`
- `evidence_index`
- `agent_started`
- `agent_token`
- `agent_complete`
- `complete`
- `error`
- `vision_started`
- `vision_complete`
- `vision_error`

The React UI updates each Agent card, displays token-level raw model generation in a compact live preview, and renders parsed structured results when each Agent completes.

## Test Commands

Preferred:

```bash
python -m pytest -q
```

Fallback:

```bash
python -m unittest tests.test_pdf_parser tests.test_llm tests.test_evidence -v
```

At inspection time, tests could not run in the current Codex runtime because dependencies were missing:

- `pytest` was not installed
- `fitz` / PyMuPDF was not installed

This was an environment issue, not a confirmed code failure. Install `requirements.txt` before treating test failures as meaningful.

## Architecture Notes

### Schemas

All structured outputs live in `core/schemas.py`.

Models:

- `EvidenceItem`
- `MethodOutput`
- `ExperimentOutput`
- `CriticOutput`
- `SummaryOutput`

Each agent output may include `evidence`, a list of snippet IDs, page labels, short quotes/paraphrases, and support notes. When changing output shape, update all affected prompts, agents, summary synthesis, CLI assumptions, frontend rendering, and tests.

### Evidence Index

Primary file: `core/evidence.py`

Public API:

- `build_evidence_index(parsed_paper) -> list[EvidenceSnippet]`
- `evidence_context_for_agent(snippets, agent_type) -> str`
- `evidence_payload(snippets) -> list[dict]`

This is not summary compression. It keeps bounded slices of original paper evidence with stable IDs, section names, and page labels. Text evidence uses IDs such as `E003`, extracted table evidence uses `T001`, and figure/caption or vision-model visual-summary evidence uses `F001`. Agents receive these snippets and are prompted to cite only IDs that appear in their input.

Current limitations:

- Table evidence is extracted locally with PyMuPDF `find_tables()` when the PDF exposes a usable table layout. Complex scanned tables may still need OCR or a vision model.
- Figure evidence renders detected visual regions or full pages to PNG and can call the configured vision model to produce concise visual summaries before agent analysis. Vision enrichment fans out concurrent requests by default: `VISION_MAX_FIGURES=0` means all visual candidates, and `VISION_MAX_WORKERS=0` means one worker per selected figure. If the provider returns rate-limit errors such as 429/1302, failed figures are retried with `VISION_RETRY_WORKERS` and `VISION_RATE_LIMIT_RETRIES`. If no vision model is configured or a figure call ultimately fails, text/table/caption evidence still proceeds.

### PDF Parser

Primary file: `core/pdf_parser.py`

Public API:

- `parse_pdf(pdf_path) -> ParsedPaper`
- `ParsedPaper.get_sections_for_agent(agent_type) -> str`

Parsing strategy:

1. Open PDF with PyMuPDF (`fitz`).
2. Extract metadata and page text.
3. Extract best-effort tables and figure/caption signals.
4. For live web analysis, optionally render visual regions and call `VISION_MODEL_NAME` to attach `FigureBlock.visual_summary`.
5. Prefer embedded PDF outline/bookmarks for section splitting.
6. If too few sections are found, try font-size-based section splitting.
7. If still too few sections are found, fall back to regex-based headings.
8. If still too few sections are found, return one `Full Paper` section.

Section routing is keyword-based and supports English and Chinese headings.

Agent routing:

- `method`: abstract, introduction, related work, background, method/model/framework/system/architecture/proposed, and Chinese equivalents
- `experiment`: abstract, experiment/results/discussion/analysis/evaluation, and Chinese equivalents
- `critic`: abstract, introduction, related work, conclusion/limitations/future work/discussion, and Chinese equivalents

If no matching sections are found for an agent, the full paper text is returned.

### Agents

Each agent is intentionally thin:

1. Read the corresponding prompt from `prompts/`.
2. Replace placeholders.
3. Call either `invoke_structured_with_retry(...)` for non-streaming use or `stream_structured_with_retry(...)` for token-level streaming.
4. Return the relevant Pydantic model.

Files:

- `agents/method_agent.py`
- `agents/experiment_agent.py`
- `agents/critic_agent.py`
- `agents/summary_agent.py`

Prompt placeholders:

- `method.txt`: `{paper_text}`
- `experiment.txt`: `{paper_text}`
- `critic.txt`: `{paper_text}`
- `summary.txt`: `{paper_title}`, `{method_output}`, `{experiment_output}`, `{critic_output}`

### Graph

Primary file: `core/graph.py`

The LangGraph workflow is fan-out/fan-in:

```text
START
-> evidence
-> method
-> experiment
-> critic
method + experiment + critic
-> summary
-> END
```

`run_pipeline(parsed_paper)` builds and compiles the graph on each call. This is fine for CLI use. If building a server or batch runner, consider caching the compiled graph.

### LLM Wrapper

Primary file: `utils/llm.py`

Important behavior:

- Loads `.env` from repo root with `override=False`.
- Caches `ChatOpenAI` through `@lru_cache(maxsize=1)`.
- Raises `EnvironmentError` if no GLM/OpenAI-compatible API key is configured.
- Uses both OpenAI client retry (`max_retries=3`) and local `invoke_with_retry`.
- `stream_structured_with_retry(...)` streams raw JSON tokens via `ChatOpenAI.stream(...)`, forwards each token through a callback, then parses the accumulated output into the requested Pydantic schema.

Provider compatibility depends on `langchain-openai` structured-output support for the configured endpoint.

## Development Rules For Codex

Prefer small, local changes that preserve the existing layering:

1. `core/schemas.py` defines contracts.
2. `prompts/*.txt` define agent instructions.
3. `agents/*.py` wire prompts to structured LLM calls.
4. `core/pdf_parser.py` controls input extraction and section routing.
5. `core/graph.py` controls orchestration.
6. `main.py` controls the CLI surface.
7. `app.py` controls the web API surface.
8. `frontend-prototype/` controls the React workbench.

Do not silently change the CLI contract. Tests currently check that the PDF path is positional, not `--pdf`.

Use UTF-8 when reading and writing project docs and prompts. Existing docs are Chinese/English UTF-8; PowerShell default encoding may display Chinese as mojibake unless `-Encoding UTF8` is used.

Do not commit `.env`, `.venv`, PDFs under `examples/`, build artifacts, or caches.

When adding a new agent, update:

- schema
- prompt
- agent wrapper
- graph state
- graph nodes and edges
- summary prompt inputs
- tests

When changing parser behavior, update parser tests first or alongside the change.

When touching LLM calls, add mocked tests where possible. Avoid tests that require a real API key.

## Known Gaps

High-value missing pieces:

- No lockfile or pinned dependency set.
- No real PDF fixture test.
- No mocked tests for agent prompt/schema wiring.
- No graph-level integration test.
- No CLI subprocess test.
- No examples despite README using `examples/your_paper.pdf`.
- No chunking/truncation guard for long papers.

Runtime risks:

- Long papers may exceed model context.
- Section splitting is heuristic and may misclassify unusual PDF layouts.
- Regex fallback for Chinese short headings is broad and may over-detect in some papers.
- Structured output behavior may vary across OpenAI-compatible providers.

## Good Next Changes

If improving setup:

1. Add `LLM_TEMPERATURE=1.0` to `.env.example`.
2. Add a dependency lock strategy.
3. Add a short `docs/` page or make `CLAUDE.md` match the real tree.

If improving quality:

1. Add a tiny generated PDF fixture for parser tests.
2. Add mocked agent tests.
3. Add graph integration tests with fake agent outputs.
4. Add CLI subprocess tests for missing-file and output-file behavior.

If improving product behavior:

1. Add chunking or summarization before LLM calls.
2. Add better PDF title extraction fallback.
3. Add token-level streaming only if the structured-output contract is changed to support incremental drafts safely.
4. Add progress/error reporting around individual agent failures.
