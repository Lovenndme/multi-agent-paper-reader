"""Tests for provider-neutral Agentic RAG tools, loops, and checkpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from core.agentic_checkpoints import (
    load_agentic_checkpoints,
    save_agentic_checkpoint,
)
from core.agentic_runtime import (
    AgenticRetrievalRequest,
    AgenticRetrievalRuntime,
)
from core.agentic_types import AgenticRunBudget, RetrievalAction
from core.evidence import EvidenceSnippet
from core.model_providers import PROVIDERS, provider_agentic_capability
from core.paper_tools import (
    PaperToolError,
    PaperToolRegistry,
    safe_calculate,
    search_paper_evidence,
)


def _snippets() -> tuple[EvidenceSnippet, ...]:
    return (
        EvidenceSnippet(
            "E001",
            "Method",
            1,
            1,
            "The encoder uses a gated attention architecture.",
        ),
        EvidenceSnippet(
            "T001",
            "Experiments",
            4,
            4,
            "Accuracy | Proposed | 91.2",
            "table",
        ),
        EvidenceSnippet(
            "F001",
            "Architecture",
            2,
            2,
            "Vision summary: encoder to decoder pipeline.",
            "figure",
        ),
    )


def test_all_providers_have_structured_action_fallback() -> None:
    assert set(PROVIDERS) <= {
        provider_id
        for provider_id in PROVIDERS
        if provider_agentic_capability(provider_id).structured_actions
    }
    assert provider_agentic_capability("openai").native_tool_calling is True
    assert provider_agentic_capability("custom").native_tool_calling is False


def test_paper_search_is_fresh_hybrid_retrieval_with_kind_filter() -> None:
    snippets = _snippets()
    with patch(
        "core.paper_tools.semantic_scores",
        return_value=[0.2, 0.95, 0.1],
    ):
        selected = search_paper_evidence(
            snippets,
            "quantitative accuracy result table",
            limit=2,
        )
    assert selected[0].id == "T001"

    with patch("core.paper_tools.semantic_scores", return_value=[0.8]):
        figures = search_paper_evidence(
            snippets,
            "architecture",
            kind="figure",
            limit=2,
        )
    assert [item.id for item in figures] == ["F001"]


def test_read_only_registry_and_calculator_reject_unsafe_inputs() -> None:
    registry = PaperToolRegistry.create(_snippets(), title="Paper")
    overview = registry.execute(
        RetrievalAction(
            tool="paper_overview",
            public_summary="查看论文结构。",
        )
    )
    assert "tables: 1" in overview.content
    assert safe_calculate("sqrt(16) + 2**3") == 12
    with pytest.raises(PaperToolError):
        safe_calculate("__import__('os').system('id')")
    with pytest.raises(PaperToolError):
        registry.execute(
            RetrievalAction(
                tool="paper_read_table",
                evidence_id="T999",
                public_summary="读取表格。",
            )
        )
    assert {tool.name for tool in registry.native_tools()} == {
        "paper_search",
        "paper_overview",
        "paper_read_section",
        "paper_read_page",
        "paper_read_table",
        "paper_read_figure",
        "calculate",
        "finish_retrieval",
    }


def test_structured_action_loop_collects_evidence_and_stops() -> None:
    runtime = AgenticRetrievalRuntime()
    events: list[str] = []
    actions = [
        RetrievalAction(
            tool="paper_search",
            query="accuracy result",
            public_summary="正在检索实验结果。",
        ),
        RetrievalAction(
            tool="finish_retrieval",
            public_summary="实验结果证据已足够。",
        ),
    ]
    with (
        patch.dict(
            "os.environ",
            {"RAG_MODE": "agentic", "AGENTIC_TOOL_STRATEGY": "structured"},
        ),
        patch.object(runtime, "_structured_action", side_effect=actions),
        patch("core.paper_tools.semantic_scores", return_value=[0.1, 0.9, 0.2]),
        patch("core.agentic_runtime.save_agentic_checkpoint"),
        patch("core.agentic_runtime.get_llm") as get_llm_mock,
    ):
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="experiment",
                objective="Verify reported accuracy.",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                emit=lambda event_type, _payload: events.append(event_type),
            )
        )

    assert {item.id for item in result.snippets} >= {"E001", "T001"}
    assert result.stop_reason == "model_finished"
    assert result.strategy == "structured"
    assert [step.action.tool for step in result.steps] == [
        "paper_search",
        "finish_retrieval",
    ]
    assert events[0] == "retrieval_started"
    assert "tool_complete" in events
    assert events[-1] == "retrieval_complete"
    get_llm_mock.assert_not_called()


def test_native_failure_falls_back_without_losing_seed_evidence() -> None:
    class FakeNativeModel:
        def bind_tools(self, _tools):
            return self

    runtime = AgenticRetrievalRuntime()
    finish = RetrievalAction(
        tool="finish_retrieval",
        public_summary="预选证据已足够。",
    )
    with (
        patch.dict(
            "os.environ",
            {"RAG_MODE": "agentic", "AGENTIC_TOOL_STRATEGY": "auto"},
        ),
        patch.object(runtime, "_native_action", side_effect=RuntimeError("unsupported")),
        patch.object(runtime, "_structured_action", return_value=finish),
        patch("core.agentic_runtime.save_agentic_checkpoint"),
    ):
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="method",
                objective="Check architecture.",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                provider_id="openai",
                llm=FakeNativeModel(),
            )
        )

    assert result.fallback_used is True
    assert result.strategy == "structured"
    assert [item.id for item in result.snippets] == ["E001"]


def test_native_tool_call_is_parsed_into_the_same_action_contract() -> None:
    class FakeNativeModel:
        def bind_tools(self, tools):
            assert any(tool.name == "paper_search" for tool in tools)
            return self

    runtime = AgenticRetrievalRuntime()
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "paper_search",
                "args": {
                    "query": "gated attention architecture",
                    "kind": "text",
                    "limit": 3,
                    "public_summary": "正在核对架构证据。",
                },
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    request = AgenticRetrievalRequest(
        agent_id="method",
        objective="Verify architecture.",
        registry=PaperToolRegistry.create(_snippets()),
        provider_id="openai",
        llm=FakeNativeModel(),
    )
    with patch("core.agentic_runtime.invoke_with_retry", return_value=response):
        action = runtime._native_action(
            request,
            (),
            (),
            request.budget or AgenticRunBudget(),
        )

    assert action.tool == "paper_search"
    assert action.query == "gated attention architecture"


def test_hybrid_flag_skips_model_driven_loop() -> None:
    runtime = AgenticRetrievalRuntime()
    with patch.dict("os.environ", {"RAG_MODE": "hybrid"}):
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="critic",
                objective="Check limitations.",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
            )
        )
    assert result.strategy == "hybrid"
    assert result.steps == ()


def test_public_checkpoints_persist_without_prompts(
    tmp_path: Path,
) -> None:
    payload = {
        "run_id": "run-test",
        "agent": "method",
        "step": 1,
        "tool": "paper_search",
        "summary": "已完成检索。",
        "evidence_count": 2,
        "private_prompt": "must not persist",
    }
    with patch.dict(
        "os.environ",
        {
            "PAPER_READER_DATA_DIR": str(tmp_path),
            "AGENTIC_RAG_CHECKPOINTS": "true",
        },
    ):
        save_agentic_checkpoint("tool_complete", payload)
        restored = load_agentic_checkpoints("run-test")

    assert restored[0]["state"]["summary"] == "已完成检索。"
    assert "private_prompt" not in restored[0]["state"]
