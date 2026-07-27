"""Tests for provider-neutral Agentic RAG tools, loops, and checkpoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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
from core.agentic_types import (
    AgenticRagConfig,
    AgenticRunBudget,
    RetrievalAction,
)
from core.evidence import EvidenceSnippet
from core.model_providers import PROVIDERS, provider_agentic_capability
from core.paper_tools import (
    PaperToolError,
    PaperToolRegistry,
    prefixed_snippets,
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
        "core.hybrid_retrieval.semantic_scores",
        return_value=[0.2, 0.95, 0.1],
    ):
        selected = search_paper_evidence(
            snippets,
            "quantitative accuracy result table",
            limit=2,
        )
    assert selected[0].id == "T001"

    with patch("core.hybrid_retrieval.semantic_scores", return_value=[0.8]):
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
            patch(
                "core.hybrid_retrieval.semantic_scores",
                return_value=[0.1, 0.9, 0.2],
            ),
        patch("core.agentic_runtime.save_agentic_checkpoint"),
        patch(
            "core.agentic_runtime.get_agentic_planner_llm_for_route"
        ) as planner_llm_mock,
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
    planner_llm_mock.assert_not_called()


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


def test_adaptive_skip_policy_uses_seed_without_initializing_planner() -> None:
    runtime = AgenticRetrievalRuntime()
    config = AgenticRagConfig(mode="adaptive", tool_strategy="structured")
    with (
        patch.dict("os.environ", {"RAG_MODE": "agentic"}),
        patch.object(runtime, "_structured_action") as planner,
        patch("core.agentic_runtime.get_agentic_planner_llm_for_route") as get_planner,
    ):
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="method",
                objective="Verify the method.",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                policy="skip",
                config=config,
            )
        )

    assert result.strategy == "adaptive_static"
    assert result.stop_reason == "adaptive_policy_skip"
    assert result.adaptive_triggered is False
    assert result.mode == "adaptive"
    assert result.policy == "skip"
    planner.assert_not_called()
    get_planner.assert_not_called()


def test_hybrid_overrides_force_while_agentic_overrides_skip() -> None:
    runtime = AgenticRetrievalRuntime()
    finish = RetrievalAction(
        tool="finish_retrieval",
        public_summary="证据已足够。",
    )
    with patch.object(runtime, "_structured_action", return_value=finish) as planner:
        hybrid = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="method",
                objective="Verify the method.",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                policy="force",
                config=AgenticRagConfig(
                    mode="hybrid",
                    tool_strategy="structured",
                ),
            )
        )
        agentic = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="method",
                objective="Verify the method.",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                policy="skip",
                config=AgenticRagConfig(
                    mode="agentic",
                    tool_strategy="structured",
                ),
            )
        )

    assert hybrid.strategy == "hybrid"
    assert hybrid.steps == ()
    assert agentic.stop_reason == "model_finished"
    assert agentic.mode == "agentic"
    assert planner.call_count == 1


def test_adaptive_force_is_bounded_to_two_planner_steps() -> None:
    runtime = AgenticRetrievalRuntime()
    actions = [
        RetrievalAction(
            tool="paper_overview",
            public_summary="查看论文结构。",
        ),
        RetrievalAction(
            tool="paper_read_table",
            evidence_id="T001",
            public_summary="核对实验表格。",
        ),
    ]
    with (
        patch.object(runtime, "_structured_action", side_effect=actions) as planner,
        patch("core.agentic_runtime.save_agentic_checkpoint"),
    ):
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="experiment",
                objective="Repair missing experiment evidence.",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                policy="force",
                budget=AgenticRunBudget(max_steps=4),
                config=AgenticRagConfig(
                    mode="adaptive",
                    tool_strategy="structured",
                    adaptive_max_steps=2,
                ),
            )
        )

    assert planner.call_count == 2
    assert len(result.steps) == 2
    assert result.stop_reason == "budget_exhausted"
    assert result.adaptive_triggered is True
    assert result.adaptive_reason == "policy_force"


def test_adaptive_chat_gate_skips_simple_question_and_triggers_numeric_gap() -> None:
    seed = (_snippets()[0],)
    config = AgenticRagConfig(mode="adaptive", tool_strategy="structured")
    simple_runtime = AgenticRetrievalRuntime()
    with patch.object(simple_runtime, "_structured_action") as simple_planner:
        simple = simple_runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="paper_chat",
                objective="Answer the follow-up question.",
                adaptive_query="Explain gated attention.",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=seed,
                config=config,
            )
        )

    numeric_runtime = AgenticRetrievalRuntime()
    finish = RetrievalAction(
        tool="finish_retrieval",
        public_summary="数值证据核对完成。",
    )
    with patch.object(
        numeric_runtime,
        "_structured_action",
        return_value=finish,
    ) as numeric_planner:
        numeric = numeric_runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="paper_chat",
                objective="Answer the follow-up question.",
                adaptive_query="What exact accuracy score is reported?",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=seed,
                config=config,
            )
        )

    assert simple.adaptive_triggered is False
    assert simple.adaptive_reason == "query_seed_sufficient"
    simple_planner.assert_not_called()
    assert numeric.adaptive_triggered is True
    assert numeric.adaptive_reason == "exact_numeric_verification"
    assert numeric_planner.call_count == 1


@pytest.mark.parametrize(
    "question",
    (
        "这个方法是什么？",
        "请概括这篇论文。",
        "什么是注意力机制？",
    ),
)
def test_adaptive_gate_keeps_simple_chinese_questions_static(question: str) -> None:
    runtime = AgenticRetrievalRuntime()
    with patch.object(runtime, "_structured_action") as planner:
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="paper_chat",
                objective="Answer the follow-up question.",
                adaptive_query=question,
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                config=AgenticRagConfig(
                    mode="adaptive",
                    tool_strategy="structured",
                ),
            )
        )

    assert result.adaptive_triggered is False
    assert result.adaptive_reason == "cjk_simple_question"
    planner.assert_not_called()


def test_adaptive_gate_retrieves_for_substantive_low_coverage_chinese_question() -> None:
    finish = RetrievalAction(
        tool="finish_retrieval",
        public_summary="已核对低资源场景下的失败模式。",
    )
    runtime = AgenticRetrievalRuntime()
    with patch.object(runtime, "_structured_action", return_value=finish) as planner:
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="paper_chat",
                objective="Answer the follow-up question.",
                adaptive_query="论文在低资源场景下有哪些失败模式和局限性？",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                config=AgenticRagConfig(
                    mode="adaptive",
                    tool_strategy="structured",
                ),
            )
        )

    assert result.adaptive_triggered is True
    assert result.adaptive_reason in {
        "low_query_coverage",
        "low_retrieval_confidence",
    }
    assert planner.call_count == 1


def test_adaptive_gate_uses_dense_bm25_disagreement_as_low_confidence() -> None:
    finish = RetrievalAction(
        tool="finish_retrieval",
        public_summary="已补查低置信度证据。",
    )
    runtime = AgenticRetrievalRuntime()
    confidence = SimpleNamespace(
        low_confidence=True,
        dense_bm25_overlap_at_5=0,
        score_margin=0.0001,
    )
    with (
        patch.object(runtime, "_structured_action", return_value=finish) as planner,
        patch(
            "core.agentic_runtime.rank_evidence",
            return_value=SimpleNamespace(diagnostics=confidence),
        ),
    ):
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="paper_chat",
                objective="Answer the follow-up question.",
                adaptive_query=(
                    "Explain the unusual training failure under sparse supervision."
                ),
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                config=AgenticRagConfig(
                    mode="adaptive",
                    tool_strategy="structured",
                ),
            )
        )

    assert result.adaptive_triggered is True
    assert result.adaptive_reason == "low_retrieval_confidence"
    assert planner.call_count == 1


def test_exact_numeric_question_ignores_unrelated_year_in_seed() -> None:
    seed = EvidenceSnippet(
        "E001",
        "Experiments",
        1,
        1,
        "A 2024 evaluation discusses the accuracy metric but omits its exact value.",
    )
    finish = RetrievalAction(
        tool="finish_retrieval",
        public_summary="精确数值核对完成。",
    )
    runtime = AgenticRetrievalRuntime()
    with patch.object(runtime, "_structured_action", return_value=finish) as planner:
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="paper_chat",
                objective="Answer the follow-up question.",
                adaptive_query="What exact accuracy score is reported?",
                registry=PaperToolRegistry.create((seed,)),
                seed_snippets=(seed,),
                config=AgenticRagConfig(
                    mode="adaptive",
                    tool_strategy="structured",
                ),
            )
        )

    assert result.adaptive_triggered is True
    assert result.adaptive_reason == "exact_numeric_verification"
    assert planner.call_count == 1


def test_explicit_evidence_id_does_not_bypass_exact_numeric_verification() -> None:
    seed = EvidenceSnippet(
        "E001",
        "Experiments",
        1,
        1,
        "A 2024 evaluation discusses accuracy but omits its exact value.",
    )
    finish = RetrievalAction(
        tool="finish_retrieval",
        public_summary="已核对指定证据对应的准确率。",
    )
    runtime = AgenticRetrievalRuntime()
    with patch.object(runtime, "_structured_action", return_value=finish) as planner:
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="paper_chat",
                objective="Answer the follow-up question.",
                adaptive_query="请解释 E001 中准确率的具体数值是多少？",
                registry=PaperToolRegistry.create((seed,)),
                seed_snippets=(seed,),
                config=AgenticRagConfig(
                    mode="adaptive",
                    tool_strategy="structured",
                ),
            )
        )

    assert result.adaptive_triggered is True
    assert result.adaptive_reason == "exact_numeric_verification"
    assert planner.call_count == 1


def test_planner_factory_failure_preserves_deterministic_seed() -> None:
    runtime = AgenticRetrievalRuntime()
    with patch(
        "core.agentic_runtime.get_agentic_planner_llm_for_route",
        side_effect=RuntimeError("planner route unavailable"),
    ):
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="method",
                objective="Verify the method.",
                registry=PaperToolRegistry.create(_snippets()),
                seed_snippets=(_snippets()[0],),
                provider_id="openai",
                model="gpt-5.6-sol",
                model_mode="medium",
                config=AgenticRagConfig(
                    mode="agentic",
                    tool_strategy="auto",
                ),
            )
        )

    assert [item.id for item in result.snippets] == ["E001"]
    assert result.fallback_used is True
    assert result.stop_reason == "planner_fallback:RuntimeError"


def test_multi_paper_seed_is_round_robin_bounded_without_starving_sources() -> None:
    groups = [
        (
            f"P{paper_index}",
            tuple(
                EvidenceSnippet(
                    f"E{item_index:03d}",
                    "Method",
                    item_index,
                    item_index,
                    f"paper {paper_index} evidence {item_index}",
                )
                for item_index in range(1, 7)
            ),
        )
        for paper_index in range(1, 5)
    ]
    balanced = prefixed_snippets(groups)
    result = AgenticRetrievalRuntime().retrieve(
        AgenticRetrievalRequest(
            agent_id="comparison_chat",
            objective="Summarize the selected evidence.",
            registry=PaperToolRegistry.create(balanced),
            seed_snippets=balanced,
            budget=AgenticRunBudget(max_evidence_items=16),
            config=AgenticRagConfig(mode="hybrid"),
        )
    )

    labels = [item.id.split(":", 1)[0] for item in result.snippets]
    assert len(result.snippets) == 16
    assert {label: labels.count(label) for label in set(labels)} == {
        "P1": 4,
        "P2": 4,
        "P3": 4,
        "P4": 4,
    }


def test_oversized_first_source_respects_char_budget_and_preserves_all_sources() -> None:
    snippets = tuple(
        EvidenceSnippet(
            f"P{paper_index}:E001",
            "Results",
            1,
            1,
            "x" * (25_000 if paper_index == 1 else 2_000),
        )
        for paper_index in range(1, 5)
    )
    result = AgenticRetrievalRuntime().retrieve(
        AgenticRetrievalRequest(
            agent_id="comparison_chat",
            objective="Compare the selected evidence.",
            registry=PaperToolRegistry.create(snippets),
            seed_snippets=snippets,
            budget=AgenticRunBudget(
                max_evidence_items=16,
                max_evidence_chars=24_000,
            ),
            config=AgenticRagConfig(mode="hybrid"),
        )
    )

    assert sum(len(item.text) for item in result.snippets) <= 24_000
    assert {item.id.split(":", 1)[0] for item in result.snippets} == {
        "P1",
        "P2",
        "P3",
        "P4",
    }
    assert len(result.snippets[0].text) == 6_000


def test_new_tool_evidence_can_replace_a_full_seed_set() -> None:
    snippets = tuple(
        EvidenceSnippet(
            f"E{index:03d}",
            "Body",
            index - 1,
            index - 1,
            f"evidence on page {index}",
        )
        for index in range(1, 18)
    )
    actions = [
        RetrievalAction(
            tool="paper_read_page",
            page=17,
            public_summary="读取新页面证据。",
        ),
        RetrievalAction(
            tool="finish_retrieval",
            public_summary="新增证据已足够。",
        ),
    ]
    runtime = AgenticRetrievalRuntime()
    with patch.object(runtime, "_structured_action", side_effect=actions):
        result = runtime.retrieve(
            AgenticRetrievalRequest(
                agent_id="method",
                objective="Read missing evidence.",
                registry=PaperToolRegistry.create(snippets),
                seed_snippets=snippets[:16],
                budget=AgenticRunBudget(
                    max_steps=2,
                    max_evidence_items=16,
                ),
                config=AgenticRagConfig(
                    mode="agentic",
                    tool_strategy="structured",
                ),
            )
        )

    assert len(result.snippets) == 16
    assert "E017" in {item.id for item in result.snippets}


def test_adaptive_trace_exposes_frozen_mode_policy_and_reason() -> None:
    result = AgenticRetrievalRuntime().retrieve(
        AgenticRetrievalRequest(
            agent_id="summary",
            objective="Synthesize cited evidence.",
            registry=PaperToolRegistry.create(_snippets()),
            seed_snippets=(_snippets()[0],),
            policy="skip",
            config=AgenticRagConfig(mode="adaptive"),
        )
    )

    assert result.public_trace() == {
        "strategy": "adaptive_static",
        "stop_reason": "adaptive_policy_skip",
        "fallback_used": False,
        "evidence_ids": ["E001"],
        "steps": [],
        "mode": "adaptive",
        "policy": "skip",
        "adaptive_triggered": False,
        "adaptive_reason": "policy_skip",
    }


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
