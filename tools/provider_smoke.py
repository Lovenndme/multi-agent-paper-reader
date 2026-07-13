"""Run secret-safe live routing and grounded paper-QA smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.chat import PaperChatRequest, build_chat_prompt, store_analysis_session, stream_chat_reply
from core.evidence import EvidenceSnippet
from core.model_providers import PROVIDERS, provider_api_key
from utils.llm import (
    get_llm,
    reset_llm_clients,
    start_text_model_call_trace,
    update_text_model_call_trace,
)


DEFAULT_PROVIDERS = ("openai", "deepseek", "doubao")


def run_provider_smoke(provider_id: str) -> dict[str, Any]:
    spec = PROVIDERS[provider_id]
    if not provider_api_key(provider_id):
        return {
            "provider": provider_id,
            "status": "missing_key",
            "requested_model": spec.default_text_model,
        }

    old_values = {
        name: os.environ.get(name)
        for name in ("TEXT_PROVIDER", "MODEL_NAME", "ENABLE_VISION_SUMMARY")
    }
    os.environ.update(
        {
            "TEXT_PROVIDER": provider_id,
            "MODEL_NAME": spec.default_text_model,
            "ENABLE_VISION_SUMMARY": "false",
        }
    )
    reset_llm_clients()
    try:
        llm = get_llm()
        minimal_trace = start_text_model_call_trace(llm)
        minimal_response = llm.invoke([HumanMessage(content="只回复 OK")])
        update_text_model_call_trace(minimal_trace, minimal_response)
        if not getattr(minimal_response, "content", None):
            raise RuntimeError("minimal request returned no content")

        snippet = EvidenceSnippet(
            id="E001",
            section="Experiments",
            page_start=0,
            page_end=0,
            text=(
                "On the held-out benchmark, PaperReader achieved 91.2% accuracy, "
                "exceeding the strongest baseline by 4.8 percentage points."
            ),
        )
        context = {
            "mode": "live",
            "paper": {"title": "PaperReader Provider Smoke Test"},
            "evidence_index": [
                {
                    "id": "E001",
                    "section": "Experiments",
                    "page": "p.1",
                    "quote": snippet.text,
                }
            ],
        }
        analysis_id = store_analysis_session([snippet], context)
        request = PaperChatRequest(
            analysis_id=analysis_id,
            question="这篇论文的核心实验结果是什么？请引用论文证据。",
            context=context,
        )
        prompt = build_chat_prompt(request)
        qa_trace: dict[str, Any] = {}
        answer = "".join(
            stream_chat_reply(request, messages=prompt.messages, trace=qa_trace)
        )
        if "E001" not in answer:
            raise RuntimeError("paper QA response did not cite E001")

        return {
            "provider": provider_id,
            "status": "passed",
            "requested_model": qa_trace.get("requested_model"),
            "upstream_model": qa_trace.get("upstream_model"),
            "endpoint_host": qa_trace.get("endpoint_host"),
            "request_id": qa_trace.get("request_id"),
            "verification": qa_trace.get("verification"),
            "paper_qa_cited_evidence": True,
            "minimal_request_verified": bool(
                minimal_trace.get("request_id") or minimal_trace.get("upstream_model")
            ),
        }
    except Exception as exc:  # noqa: BLE001 - CLI reports only safe error type
        return {
            "provider": provider_id,
            "status": "failed",
            "requested_model": spec.default_text_model,
            "error_type": type(exc).__name__,
        }
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_llm_clients()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "providers",
        nargs="*",
        choices=tuple(PROVIDERS),
        default=list(DEFAULT_PROVIDERS),
    )
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    results = [run_provider_smoke(provider_id) for provider_id in args.providers]
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    if any(result["status"] == "failed" for result in results):
        return 1
    if any(result["status"] == "missing_key" for result in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
