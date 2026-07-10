# Multi-Agent Paper Reader

Multi-Agent Paper Reader is an evidence-grounded research paper reading assistant. Upload a PDF, parse its sections, build a traceable evidence index from text, extracted tables, and vision-model figure summaries, run specialist agents for method analysis, experiment analysis, and critique, then synthesize a polished structured reading note.

## Web App

The repository includes a full-stack web app:

- Backend: `app.py` with FastAPI
- Frontend: `frontend-prototype/` with React + Vite
- API: `POST /api/analyze` accepts a PDF upload and returns parsed paper metadata plus all agent outputs
- Streaming API: `POST /api/analyze/stream` returns newline-delimited JSON events for parsing, evidence indexing, token-level model output, agent completion, and final synthesis
- Follow-up API: `POST /api/chat/stream` streams analysis-grounded answers using the selected text, current Agent outputs, evidence previews, and recent conversation turns
- Section titles: common headings use a local Chinese dictionary; unknown English headings are translated in one bounded GLM batch before Live analysis starts
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

For real LLM analysis, copy `.env.example` to `.env` and set `GLM_API_KEY`. The default provider is Zhipu GLM at `https://open.bigmodel.cn/api/paas/v4`, using `glm-5.2`. The UI also has a Demo mode in Model Settings so upload, PDF parsing, API wiring, and result rendering can be verified without an LLM key.

For figure/chart understanding, set `ENABLE_VISION_SUMMARY=true` and `VISION_MODEL_NAME=glm-5v-turbo` or another OpenAI-compatible vision model. The backend renders PDF visual regions to PNG, fans out one concurrent vision request per selected figure/chart by default, asks the vision model for concise Chinese visual summaries, and indexes them as `F` evidence. If the provider returns rate-limit errors, failed figures are automatically retried with the smaller `VISION_RETRY_WORKERS` pool.

## CLI Quick Start

```bash
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set GLM_API_KEY

python main.py examples/your_paper.pdf --pretty
```

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

## Explainable Assessment

Every completed API response includes an `assessment` object with two separate results:

- **Novelty (1-5):** the Critic Agent scores problem originality (15%), method originality (40%), difference from prior work (30%), and generality (15%). The backend calculates the weighted total and keeps each reason and supporting evidence ID.
- **Analysis reliability (0-100):** the backend deterministically scores PDF parsing (20), key-section coverage (35), valid evidence citations (30), and structured-output integrity (15).

Reliability is not the model's self-reported confidence. Missing related-work coverage, fewer than three valid citations, insufficient parsed content, incomplete novelty dimensions, or Demo mode apply explicit score caps. The response exposes the component scores, raw score, cap, final score, and warnings so the result can be audited.

## Follow-up Chat

Select text inside the results panel and choose **在侧边聊天中提问** to open the paper chat drawer. A Live analysis continues with the configured `glm-5.2` model and sends bounded context from the current paper, Agent outputs, assessment, evidence previews, selected excerpt, and up to 16 recent conversation turns. Sample and Demo results use a deterministic reply so the complete interaction can be tested without another model call.

See [CLAUDE.md](./CLAUDE.md) for the original architecture notes.

## Tech Stack

- LangGraph for agent orchestration
- PyMuPDF for PDF parsing, outline-based section detection, table extraction, and rendered figure regions
- Pydantic v2 for output schema validation
- Evidence snippets for page/section-grounded claims (`E` text, `T` table, `F` vision figure summary)
- FastAPI for the backend API and static hosting
- React + Vite for the frontend workbench
