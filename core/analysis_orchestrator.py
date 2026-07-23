"""Application service that owns one complete paper-analysis lifecycle."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from agents.critic_agent import CRITIC_AGENT_SPEC
from agents.experiment_agent import EXPERIMENT_AGENT_SPEC
from agents.method_agent import METHOD_AGENT_SPEC
from agents.summary_agent import SUMMARY_AGENT_SPEC
from core.agent_harness import AgentHarnessError, AgentRunContext
from core.analysis_events import (
    AnalysisEvent,
    AnalysisOrchestratorError,
    AnalysisRequest,
    AnalysisResult,
    AnalysisStage,
)
from core.analysis_progress import AnalysisProgressTracker
from core.assessment import build_analysis_assessment
from core.chat import store_analysis_session
from core.codex_sdk import get_codex_sdk_service
from core.codex_tools import (
    CodexToolContextHandle,
    build_codex_paper_manifest,
    create_codex_tool_context,
)
from core.evidence import EvidenceSnippet, build_evidence_index, evidence_payload
from core.graph import GraphStageError, run_pipeline_with_state
from core.history import save_paper_analysis
from core.model_providers import (
    provider_label,
    provider_spec,
    selected_text_model,
    selected_text_model_label,
    selected_text_mode,
    selected_vision_model,
    text_provider_id,
    vision_enabled,
    vision_provider_id,
)
from core.pdf_parser import ParsedPaper, parse_pdf
from core.public_analysis import public_agent_output, public_analysis_payload
from core.schemas import CriticOutput, ExperimentOutput, MethodOutput, SummaryOutput
from core.section_titles import clean_section_title
from core.vision import enrich_paper_figures_with_vision
from utils.llm import is_llm_configured


Parser = Callable[[Path], ParsedPaper]
WorkflowRunner = Callable[..., dict[str, Any]]


def build_paper_payload(
    paper: ParsedPaper,
    filename: str,
    file_size: int,
) -> dict[str, Any]:
    """Map a parsed paper to the stable API metadata contract."""

    page_count = max((section.page_end for section in paper.sections), default=-1) + 1
    return {
        "title": paper.title,
        "filename": filename,
        "size_bytes": file_size,
        "pages": page_count,
        "sections_count": len(paper.sections),
        "sections": [
            {
                "title": section.title,
                "display_title": clean_section_title(section.title, index),
                "page_start": section.page_start,
                "page_end": section.page_end,
                "chars": len(section.content),
            }
            for index, section in enumerate(paper.sections)
        ],
        "metadata": paper.metadata,
    }


def build_model_runtime_payload() -> dict[str, Any]:
    """Describe the active route so saved analyses remain reproducible."""

    text_provider = text_provider_id()
    visual_provider = vision_provider_id()
    payload = {
        "text_provider": text_provider,
        "text_provider_label": provider_label(text_provider),
        "text_model": selected_text_model(),
        "text_model_label": selected_text_model_label(),
        "text_mode": selected_text_mode(),
        "vision_enabled": vision_enabled(),
        "vision_provider": visual_provider,
        "vision_provider_label": provider_label(visual_provider),
        "vision_model": selected_vision_model(),
    }
    if text_provider == "codex":
        payload["codex_security_profile"] = (
            get_codex_sdk_service().status().get("security_profile") or {}
        )
    return payload


def missing_model_key_message() -> str:
    """Return the user-facing setup message for the active text provider."""

    provider_id = text_provider_id()
    if provider_spec(provider_id).credential_type == "codex_login":
        return "本机 Codex 尚未登录 ChatGPT，请在 Settings 中连接 Codex 订阅。"
    return f"{provider_label(provider_id)} API Key 未配置，请在 Settings 中添加当前文本模型所需的密钥。"


def build_demo_outputs(
    paper: ParsedPaper,
    snippets: list[EvidenceSnippet] | None = None,
) -> dict[str, Any]:
    """Build deterministic schema-complete outputs without invoking a model."""

    title = paper.title or "Uploaded Paper"
    first_section = paper.sections[0].title if paper.sections else "Full Paper"
    demo_evidence = [
        {
            "id": "E001",
            "section": first_section,
            "page": "p.1",
            "quote": "Demo 模式未调用模型，仅用于验证上传、解析和渲染链路。",
            "note": "该证据说明当前输出是确定性的演示结果，不是论文内容判断。",
        }
    ]
    method = MethodOutput(
        research_problem=f"识别 {title} 所解决的核心研究问题及其主要技术路线。",
        proposed_method=(
            "Demo 模式已成功解析 PDF。配置当前文本模型的 API Key 后，可让真实 MethodAgent "
            "分析论文中与方法相关的章节。"
        ),
        key_components=[
            "PDF 解析器与章节路由",
            "MethodAgent 方法分析提示词",
            "ExperimentAgent 实验分析提示词",
            "CriticAgent 批判性评审提示词",
            "SummaryAgent 综合整理步骤",
        ],
        innovations=[
            "结构化的多 Agent 论文研读流程",
            "面向不同 Agent 的章节路由，减少无关上下文",
        ],
        differences_from_prior=(
            "该 Demo 结果用于证明前后端链路已经连通；在未运行真实 LLM 时，"
            "不会对论文的具体创新性作出判断。"
        ),
        implementation_details="在 Settings 中配置模型厂商与 API Key，即可运行真实结构化分析。",
        evidence=demo_evidence,
    )
    experiment = ExperimentOutput(
        datasets=["Demo 模式：真实数据集信息需要由 LLM 提取"],
        metrics=["Demo 模式：真实评估指标需要由 LLM 提取"],
        main_results="后端已收到并成功解析 PDF。配置模型凭证后即可提取真实实验结果。",
        comparison_with_baselines="Demo 模式不会虚构基线对比结果。",
        ablation_study=None,
        notable_findings=[
            f"解析器共识别出 {len(paper.sections)} 个章节。",
            f"提取出的正文约包含 {len(paper.full_text):,} 个字符。",
        ],
        evidence=demo_evidence,
    )
    critic = CriticOutput(
        novelty_score=3,
        novelty_justification=(
            "Demo 模式无法公正评估论文创新性，该占位结果仅用于验证完整响应结构。"
        ),
        strengths=[
            "上传、解析与响应序列化链路已经连通。",
            "前端能够渲染四个 Agent 的全部输出结构。",
        ],
        limitations=[
            "Demo 模式没有执行真实 LLM 评审。",
            "针对论文的具体批判性分析需要当前厂商的 API Key 和兼容模型。",
        ],
        potential_improvements=[
            "配置真实模型后重新运行分析。",
            "在生产环境中为超长论文增加分块处理。",
        ],
        broader_impact=None,
        evidence=demo_evidence,
    )
    summary = SummaryOutput(
        one_sentence_summary=(
            f"{title} 已成功上传并完成解析；配置真实 LLM Key 后可生成针对该论文的研读笔记。"
        ),
        core_contributions=[
            "前端上传的 PDF 已能到达 Python 后端。",
            "后端复用了项目现有的 PDF 解析器。",
            "API 返回与四 Agent 流程一致的结构化数据契约。",
            "配置模型凭证后即可运行真实 LangGraph 分析流程。",
        ],
        method_highlights=method.proposed_method,
        experiment_highlights=experiment.main_results,
        limitations_and_future_work=(
            "当前是确定性的 Demo 响应。在 Settings 中配置模型厂商与 API Key 后，"
            "即可运行真实多 Agent 分析。"
        ),
        reading_notes=(
            "Demo 模式可在不消耗模型 Token 的情况下验证部署、上传、解析和界面渲染。"
        ),
        evidence=demo_evidence,
    )
    evidence = snippets if snippets is not None else build_evidence_index(paper)
    assessment = build_analysis_assessment(
        paper,
        evidence,
        method,
        experiment,
        critic,
        summary,
        demo=True,
    )
    return {
        "method_output": method.model_dump(),
        "experiment_output": experiment.model_dump(),
        "critic_output": critic.model_dump(),
        "summary_output": summary.model_dump(),
        "assessment": assessment.model_dump(),
    }


class PaperAnalysisOrchestrator:
    """Coordinate parsing, evidence, agents, assessment, and persistence."""

    def __init__(
        self,
        *,
        parser: Parser | None = None,
        workflow_runner: WorkflowRunner | None = None,
        vision_enricher: Callable[..., Any] | None = None,
        evidence_builder: Callable[[ParsedPaper], list[EvidenceSnippet]] | None = None,
        tool_context_factory: Callable[..., CodexToolContextHandle] | None = None,
        session_store: Callable[..., str] | None = None,
        history_saver: Callable[..., str] | None = None,
        manifest_builder: Callable[[ParsedPaper], dict[str, Any]] | None = None,
        runtime_payload_builder: Callable[[], dict[str, Any]] | None = None,
        llm_configured: Callable[[], bool] | None = None,
        provider_id: Callable[[], str] | None = None,
        configuration_message: Callable[[], str] | None = None,
    ) -> None:
        self._parser = parser or parse_pdf
        self._workflow_runner = workflow_runner or run_pipeline_with_state
        self._vision_enricher = vision_enricher or enrich_paper_figures_with_vision
        self._evidence_builder = evidence_builder or build_evidence_index
        self._tool_context_factory = tool_context_factory or create_codex_tool_context
        self._session_store = session_store or store_analysis_session
        self._history_saver = history_saver or save_paper_analysis
        self._manifest_builder = manifest_builder or build_codex_paper_manifest
        self._runtime_payload_builder = runtime_payload_builder or build_model_runtime_payload
        self._llm_configured = llm_configured or is_llm_configured
        self._provider_id = provider_id or text_provider_id
        self._configuration_message = configuration_message or missing_model_key_message

    def validate(self, request: AnalysisRequest) -> None:
        """Validate conditions that must fail before an HTTP stream starts."""

        if not request.filename.lower().endswith(".pdf"):
            raise AnalysisOrchestratorError(
                "Please upload a PDF file.",
                stage=AnalysisStage.PREPARING,
                category="request",
            )
        if Path(request.filename).name != request.filename:
            raise AnalysisOrchestratorError(
                "Invalid PDF filename.",
                stage=AnalysisStage.PREPARING,
                category="request",
            )
        if not request.pdf_data:
            raise AnalysisOrchestratorError(
                "Uploaded PDF is empty.",
                stage=AnalysisStage.PREPARING,
                category="request",
            )
        if not request.demo and not self._llm_configured():
            raise AnalysisOrchestratorError(
                self._configuration_message(),
                stage=AnalysisStage.PREPARING,
                category="configuration",
            )

    def stream(self, request: AnalysisRequest) -> Iterable[AnalysisEvent]:
        """Yield the complete task lifecycle as transport-neutral events."""

        tracker = AnalysisProgressTracker()
        yield AnalysisEvent("analysis_started", tracker.started_payload())
        try:
            self.validate(request)
            with tempfile.TemporaryDirectory(prefix="paper-reader-") as tmpdir:
                pdf_path = Path(tmpdir) / request.filename
                pdf_path.write_bytes(request.pdf_data)
                try:
                    paper = self._parser(pdf_path)
                except Exception as exc:
                    raise AnalysisOrchestratorError(
                        f"Could not parse PDF: {exc}",
                        stage=AnalysisStage.PARSING,
                        category="parse",
                    ) from exc

                if request.demo:
                    yield from self._stream_demo(request, paper, tracker)
                else:
                    yield from self._stream_live(request, paper, pdf_path, tracker)
        except AnalysisOrchestratorError as exc:
            yield self._failure_event(exc, tracker)
        except Exception as exc:  # noqa: BLE001 - normalize unexpected task failures
            yield self._failure_event(
                AnalysisOrchestratorError(
                    f"Analysis failed: {exc}",
                    stage=AnalysisStage.SPECIALISTS,
                    category="runtime",
                ),
                tracker,
            )

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        """Run the same event pipeline and return its final public payload."""

        for event in self.stream(request):
            if event.type == "complete":
                return AnalysisResult(dict(event.payload))
            if event.type == "error":
                stage_value = str(event.payload.get("stage") or AnalysisStage.PREPARING.value)
                try:
                    stage = AnalysisStage(stage_value)
                except ValueError:
                    stage = AnalysisStage.PREPARING
                raise AnalysisOrchestratorError(
                    str(event.payload.get("message") or "Analysis failed."),
                    stage=stage,
                    category=str(event.payload.get("category") or "runtime"),
                    payload=dict(event.payload),
                )
        raise AnalysisOrchestratorError(
            "Analysis finished without a complete result.",
            stage=AnalysisStage.COMPLETED,
            category="incomplete",
        )

    def _stream_demo(
        self,
        request: AnalysisRequest,
        paper: ParsedPaper,
        tracker: AnalysisProgressTracker,
    ) -> Generator[AnalysisEvent, None, None]:
        paper_payload = build_paper_payload(paper, request.filename, len(request.pdf_data))
        try:
            snippets = self._evidence_builder(paper)
        except Exception as exc:
            raise AnalysisOrchestratorError(
                f"Could not build evidence index: {exc}",
                stage=AnalysisStage.EVIDENCE,
                category="evidence",
            ) from exc
        index_payload = evidence_payload(snippets)
        try:
            outputs = build_demo_outputs(paper, snippets)
        except Exception as exc:
            raise AnalysisOrchestratorError(
                f"Could not build demo assessment: {exc}",
                stage=AnalysisStage.ASSESSMENT,
                category="assessment",
            ) from exc

        yield from self._paper_and_evidence_events(
            request.mode,
            paper_payload,
            snippets,
            index_payload,
            tracker,
        )
        for spec in (METHOD_AGENT_SPEC, EXPERIMENT_AGENT_SPEC, CRITIC_AGENT_SPEC):
            yield AnalysisEvent(
                "agent_started",
                tracker.start_agent(spec.agent_id, spec.start_summary),
            )
            yield AnalysisEvent(
                "agent_progress",
                tracker.progress(
                    spec.agent_id,
                    "Demo 模式正在根据已解析的论文内容生成可验证的界面示例。",
                    progress_id=f"{spec.agent_id}-demo",
                ),
            )
            yield AnalysisEvent(
                "agent_complete",
                {
                    **tracker.complete_agent(spec.agent_id, spec.complete_summary),
                    "output_key": spec.output_key,
                    "output": public_agent_output(outputs[spec.output_key]),
                },
            )

        yield AnalysisEvent(
            "agent_started",
            tracker.start_agent("summary", SUMMARY_AGENT_SPEC.start_summary),
        )
        yield AnalysisEvent(
            "agent_complete",
            {
                **tracker.complete_agent("summary", SUMMARY_AGENT_SPEC.complete_summary),
                "output_key": "summary_output",
                "output": outputs["summary_output"],
            },
        )
        yield from self._finalize(
            request=request,
            paper=paper,
            paper_payload=paper_payload,
            snippets=snippets,
            index_payload=index_payload,
            outputs=outputs,
            tracker=tracker,
        )

    def _stream_live(
        self,
        request: AnalysisRequest,
        paper: ParsedPaper,
        pdf_path: Path,
        tracker: AnalysisProgressTracker,
    ) -> Generator[AnalysisEvent, None, None]:
        paper_payload = build_paper_payload(paper, request.filename, len(request.pdf_data))
        yield AnalysisEvent(
            "paper",
            {"mode": request.mode, "paper": paper_payload, "message": "PDF parsed"},
        )
        yield AnalysisEvent(
            "agent_progress",
            tracker.progress(
                "system",
                f"PDF 解析完成，共识别 {paper_payload['sections_count']} 个章节。",
                progress_id="paper-parsed",
            ),
        )

        yield AnalysisEvent(
            "vision_started",
            {
                "message": (
                    f"Vision enrichment started for {len(paper.figures)} visual candidates"
                )
            },
        )
        yield AnalysisEvent(
            "agent_progress",
            tracker.progress(
                "system",
                f"正在检查 {len(paper.figures)} 个图表候选区域并补充视觉证据。",
                progress_id="vision",
            ),
        )
        try:
            vision_result = self._vision_enricher(pdf_path, paper)
            yield AnalysisEvent(
                "vision_complete",
                {
                    "total_figures": vision_result.total_figures,
                    "attempted": vision_result.attempted,
                    "enriched": vision_result.enriched,
                    "skipped": vision_result.skipped,
                    "errors": vision_result.errors,
                    "message": (
                        f"Vision enrichment complete: {vision_result.enriched}/"
                        f"{vision_result.total_figures} figures enriched"
                    ),
                },
            )
            yield AnalysisEvent(
                "agent_progress",
                tracker.progress(
                    "system",
                    f"视觉检查完成，已补充 {vision_result.enriched} 个图表摘要。",
                    progress_id="vision",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - vision is an explicit soft failure
            yield AnalysisEvent(
                "vision_error",
                {"message": f"Vision enrichment failed: {exc}"},
            )
            yield AnalysisEvent(
                "agent_progress",
                tracker.progress(
                    "system",
                    "视觉摘要不可用，分析将继续使用正文、表格和图注证据。",
                    progress_id="vision",
                ),
            )

        try:
            snippets = self._evidence_builder(paper)
        except Exception as exc:
            raise AnalysisOrchestratorError(
                f"Could not build evidence index: {exc}",
                stage=AnalysisStage.EVIDENCE,
                category="evidence",
            ) from exc
        index_payload = evidence_payload(snippets)
        yield AnalysisEvent(
            "evidence_index",
            {
                "evidence_count": len(index_payload),
                "message": f"Built {len(snippets)} evidence snippets",
            },
        )
        yield AnalysisEvent(
            "agent_progress",
            tracker.progress(
                "system",
                f"已建立 {len(index_payload)} 个文本、表格或图像证据片段。",
                progress_id="evidence-index",
            ),
        )

        tool_context: CodexToolContextHandle | None = None
        try:
            if self._provider_id() == "codex":
                try:
                    tool_context = self._tool_context_factory(
                        snippets=snippets,
                        paper=paper,
                        pdf_path=pdf_path,
                    )
                except Exception as exc:
                    raise AnalysisOrchestratorError(
                        f"Could not prepare agent tools: {exc}",
                        stage=AnalysisStage.PREPARING,
                        category="tool",
                    ) from exc

            event_queue: Queue[AnalysisEvent] = Queue()

            def emit(event_type: str, payload: dict[str, Any]) -> None:
                event_queue.put(AnalysisEvent(event_type, payload))

            context = AgentRunContext(
                paper=paper,
                snippets=snippets,
                tool_context_path=tool_context.path if tool_context else None,
                tracker=tracker,
                emit=emit,
                stream=True,
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._workflow_runner,
                    paper,
                    evidence_index=snippets,
                    agent_context=context,
                )
                try:
                    final_state = yield from self._wait_for_future(future, event_queue)
                except Exception as exc:
                    harness_error = _find_harness_error(exc)
                    if harness_error is not None:
                        stage = (
                            AnalysisStage.SUMMARY
                            if harness_error.agent_id == "summary"
                            else AnalysisStage.SPECIALISTS
                        )
                        raise AnalysisOrchestratorError(
                            (
                                f"{harness_error.agent_id} failed "
                                f"({harness_error.category}): {harness_error.cause}"
                            ),
                            stage=stage,
                            category=harness_error.category,
                            payload=harness_error.failure_payload,
                        ) from exc
                    graph_error = _find_graph_stage_error(exc)
                    if graph_error is not None:
                        stage = {
                            "assessment": AnalysisStage.ASSESSMENT,
                        }.get(graph_error.stage, AnalysisStage.SPECIALISTS)
                        raise AnalysisOrchestratorError(
                            str(graph_error.cause),
                            stage=stage,
                            category=graph_error.stage,
                        ) from exc
                    raise AnalysisOrchestratorError(
                        f"Agent workflow failed: {exc}",
                        stage=AnalysisStage.SPECIALISTS,
                        category="runtime",
                    ) from exc
        finally:
            if tool_context:
                tool_context.close()

        try:
            method = MethodOutput.model_validate(final_state["method_output"])
            experiment = ExperimentOutput.model_validate(final_state["experiment_output"])
            critic = CriticOutput.model_validate(final_state["critic_output"])
        except Exception as exc:
            raise AnalysisOrchestratorError(
                f"Specialist output validation failed: {exc}",
                stage=AnalysisStage.SPECIALISTS,
                category="schema",
            ) from exc
        try:
            summary = SummaryOutput.model_validate(final_state["summary_output"])
        except Exception as exc:
            raise AnalysisOrchestratorError(
                f"Summary output validation failed: {exc}",
                stage=AnalysisStage.SUMMARY,
                category="schema",
            ) from exc
        try:
            assessment = final_state["assessment"]
            assessment_payload = (
                assessment.model_dump()
                if hasattr(assessment, "model_dump")
                else dict(assessment)
            )
        except Exception as exc:
            raise AnalysisOrchestratorError(
                f"Assessment validation failed: {exc}",
                stage=AnalysisStage.ASSESSMENT,
                category="schema",
            ) from exc
        outputs = {
            "method_output": method.model_dump(),
            "experiment_output": experiment.model_dump(),
            "critic_output": critic.model_dump(),
            "summary_output": summary.model_dump(),
            "assessment": assessment_payload,
        }
        yield from self._finalize(
            request=request,
            paper=paper,
            paper_payload=paper_payload,
            snippets=snippets,
            index_payload=index_payload,
            outputs=outputs,
            tracker=tracker,
        )

    def _paper_and_evidence_events(
        self,
        mode: str,
        paper_payload: dict[str, Any],
        snippets: list[EvidenceSnippet],
        index_payload: list[dict[str, object]],
        tracker: AnalysisProgressTracker,
    ) -> Generator[AnalysisEvent, None, None]:
        yield AnalysisEvent(
            "paper",
            {"mode": mode, "paper": paper_payload, "message": "PDF parsed"},
        )
        yield AnalysisEvent(
            "agent_progress",
            tracker.progress(
                "system",
                f"PDF 解析完成，共识别 {paper_payload['sections_count']} 个章节。",
                progress_id="paper-parsed",
            ),
        )
        yield AnalysisEvent(
            "evidence_index",
            {
                "evidence_count": len(index_payload),
                "message": f"Built {len(snippets)} evidence snippets",
            },
        )
        yield AnalysisEvent(
            "agent_progress",
            tracker.progress(
                "system",
                f"已建立 {len(index_payload)} 个文本、表格或图像证据片段。",
                progress_id="evidence-index",
            ),
        )

    def _finalize(
        self,
        *,
        request: AnalysisRequest,
        paper: ParsedPaper,
        paper_payload: dict[str, Any],
        snippets: list[EvidenceSnippet],
        index_payload: list[dict[str, object]],
        outputs: dict[str, Any],
        tracker: AnalysisProgressTracker,
    ) -> Generator[AnalysisEvent, None, None]:
        analysis_process = tracker.finish()
        try:
            runtime_payload = self._runtime_payload_builder()
        except Exception as exc:
            raise AnalysisOrchestratorError(
                f"Could not record model runtime metadata: {exc}",
                stage=AnalysisStage.PERSISTENCE,
                category="runtime_metadata",
            ) from exc
        analysis_id: str | None = None
        session_warning: str | None = None

        session_payload = {
            "mode": request.mode,
            "model_config": runtime_payload,
            "paper": paper_payload,
            "evidence_index": index_payload,
            "analysis_process": analysis_process,
            **outputs,
        }
        if not request.demo:
            try:
                analysis_id = self._session_store(snippets, session_payload)
            except Exception as exc:  # noqa: BLE001 - analysis remains usable without chat
                session_warning = f"Could not create paper chat session: {exc}"
                yield AnalysisEvent("session_error", {"message": session_warning})

        result_payload = {
            "mode": request.mode,
            "analysis_id": analysis_id,
            "model_config": runtime_payload,
            "paper": paper_payload,
            "evidence_index": index_payload,
            "analysis_process": analysis_process,
            **outputs,
        }
        history_id: str | None = None
        history_warning: str | None = None
        try:
            history_id = self._history_saver(
                pdf_data=request.pdf_data,
                result=result_payload,
                snippets=snippets,
                paper_manifest=self._manifest_builder(paper),
            )
        except Exception as exc:  # noqa: BLE001 - persistence is an explicit soft failure
            history_warning = f"Could not save paper history: {exc}"
            yield AnalysisEvent("history_error", {"message": history_warning})

        complete_payload = {
            "history_id": history_id,
            **public_analysis_payload(result_payload),
        }
        if session_warning:
            complete_payload["session_warning"] = session_warning
        if history_warning:
            complete_payload["history_warning"] = history_warning
        yield AnalysisEvent("complete", complete_payload)

    @staticmethod
    def _wait_for_future(
        future: Future[Any],
        event_queue: Queue[AnalysisEvent],
    ) -> Generator[AnalysisEvent, None, Any]:
        while not future.done():
            yield from _drain_events(event_queue)
            time.sleep(0.05)
        yield from _drain_events(event_queue)
        return future.result()

    @staticmethod
    def _failure_event(
        error: AnalysisOrchestratorError,
        tracker: AnalysisProgressTracker,
    ) -> AnalysisEvent:
        payload = dict(error.payload)
        payload.update(
            {
                "message": error.message,
                "stage": error.stage.value,
                "category": error.category,
                "analysis_process": tracker.finish(status="failed"),
            }
        )
        return AnalysisEvent("error", payload)


def _drain_events(event_queue: Queue[AnalysisEvent]) -> Generator[AnalysisEvent, None, None]:
    while True:
        try:
            yield event_queue.get_nowait()
        except Empty:
            return


def _find_harness_error(exc: BaseException) -> AgentHarnessError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, AgentHarnessError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _find_graph_stage_error(exc: BaseException) -> GraphStageError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, GraphStageError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


_DEFAULT_ORCHESTRATOR = PaperAnalysisOrchestrator()


def get_paper_analysis_orchestrator() -> PaperAnalysisOrchestrator:
    """Return the process-wide stateless paper-analysis application service."""

    return _DEFAULT_ORCHESTRATOR
