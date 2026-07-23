# Paper Analysis Orchestrator

## 定位

`core/analysis_orchestrator.py` 是整篇论文分析任务的应用服务。它位于 FastAPI
接口与 LangGraph/Agent Harness 之间：

```text
FastAPI
  -> PaperAnalysisOrchestrator
      -> PDF 解析
      -> 视觉检查（可降级）
      -> 证据索引
      -> LangGraph
          -> Harness(Method / Experiment / Critic)
          -> Harness(Summary)
          -> Assessment
      -> 分析会话与历史持久化
  -> AnalysisEvent / AnalysisResult
```

它保证一次分析只产生一个最终 `complete` 或 `error`，并在退出前清理临时 PDF
和 Codex 工具上下文。

## 类型化边界

`core/analysis_events.py` 定义三个运输无关的契约：

- `AnalysisRequest`：文件名、PDF 字节和 Demo/Live 模式；
- `AnalysisEvent`：流式生命周期事件；
- `AnalysisResult`：非流式接口返回的最终公开结果。

`PaperAnalysisOrchestrator.stream()` 是唯一执行链。`run()` 消费同一事件链并返回
最终结果，因此 `/api/analyze` 和 `/api/analyze/stream` 不再分别维护业务流程。

FastAPI 仍负责 PDF 扩展名/空文件等 HTTP 边界校验，并将领域错误映射为状态码；
模型配置、PDF 解析、Agent 编排、评估和持久化均由 Orchestrator 负责。

## 与 LangGraph、Harness、Runtime 的关系

- Orchestrator 管理整篇论文是否完整完成；
- LangGraph 表达专业 Agent 并行、Summary 汇合与 Assessment 的依赖关系；
- Harness 管理单个 Agent 的证据、公开进度、输出校验和失败状态；
- Runtime 管理单次模型调用、厂商适配、重试和工具上下文。

Orchestrator 将同一个 `AgentRunContext` 传入 LangGraph。Method、Experiment、
Critic 节点可并行运行，共享线程安全的 `AnalysisProgressTracker`，Summary 节点
等待三个结构化结果后再执行。

## 任务阶段

任务错误通过 `AnalysisStage` 标记：

```text
preparing -> parsing -> vision -> evidence -> specialists
          -> summary -> assessment -> persistence -> completed
```

公开事件继续兼容现有前端的 `analysis_started`、`agent_started`、
`agent_progress`、`agent_complete`、`history_error`、`error` 和 `complete`
协议。

## 失败策略

- PDF 解析、Agent 工作流和结构化结果失败：终止任务并发送 `error`；
- 视觉检查失败：发送 `vision_error`，继续使用正文、表格和图注；
- 分析会话创建失败：发送 `session_error`，仍返回阅读结果；
- 历史保存失败：发送 `history_error`，仍返回阅读结果；
- 无论何处失败，`analysis_process.status` 都会变为 `failed`；
- Agent 错误保留 Harness 的 `timeout`、`rate_limit`、`schema`、`tool` 或
  `runtime` 分类。

## 扩展规则

新增整篇论文阶段时，应加入 Orchestrator 或 LangGraph，而不是 FastAPI 接口：

- 与 HTTP、状态码或 NDJSON 编码有关：放在 `app.py`；
- 与完整论文任务顺序或降级策略有关：放在 Orchestrator；
- 与 Agent 依赖图有关：放在 `core/graph.py`；
- 与单个 Agent 生命周期有关：放在 Harness；
- 与模型厂商调用有关：放在 Runtime 或 `utils/llm.py`。
