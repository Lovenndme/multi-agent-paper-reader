# Agent Harness 与 Runtime

## 目标

第一版执行架构把 Agent 的“任务定义”和模型的“实际调用”分开：

```text
PaperAnalysisOrchestrator
        |
        v
     LangGraph
        |
        v
   AgentSpec
        |
        v
 AgentHarness
   |- 按 Agent 意图选择证据
   |- 生成公开生命周期事件
   |- 汇总公开过程摘要与工具活动
   |- 校验结构化输出并归类错误
        |
        v
 AgentRuntime
   |- 非流式 / 流式结构化调用
   |- 重试参数与工具上下文传递
   |- 记录 provider / model / mode / 时长
        |
        v
 utils.llm + Provider adapters
   |- Codex
   |- OpenAI-compatible
   |- Anthropic-compatible
   `- 其他已配置模型厂商
```

## 组件职责

### AgentSpec

每个 Agent 在自己的模块中声明 `AgentSpec`，包括：

- Agent ID 与 API 输出字段；
- Pydantic 输出 schema；
- Prompt 消息构造函数；
- 开始、完成、失败时的公开摘要；
- 可选的证据检索意图和证据数量/字符预算；
- 流式与非流式重试策略。

Method、Experiment、Critic、Summary 和 Comparison 都已使用同一契约。

### AgentHarness

`core/agent_harness.py` 是 Agent 的统一外层控制面。每次运行会：

1. 根据 `retrieval_profile` 从当前论文证据索引中按需选取片段；
2. 构造 Agent 消息；
3. 通过 `AnalysisProgressTracker` 发出 `agent_started`；
4. 将 Runtime 提供的公开过程摘要和工具活动转换为 `agent_progress`；
5. 调用 Runtime 并再次校验 Pydantic schema；
6. 发出带公开结果的 `agent_complete`；
7. 失败时更新 tracker，并归类为 `timeout`、`rate_limit`、`schema`、
   `tool` 或 `runtime`。

Harness 只展示模型明确提供的公开过程摘要，不暴露私有隐藏思维链，也不会把
流式结构化 JSON 当作单论文 Agent 的过程文本。内部证据 ID 在进入用户可见事件前
仍会被清理。

### AgentRuntime

`core/agent_runtime.py` 是模型执行边界。它接收 provider-neutral 的
`AgentRuntimeRequest`，并通过现有 `utils.llm` 适配层完成：

- Pydantic 结构化输出；
- 流式 token、公开过程摘要和工具活动回调；
- Codex 工具上下文传递；
- 厂商重试与流式结果修复；
- provider、model、mode、是否流式和耗时元数据记录。

Runtime 不负责论文检索、Prompt 业务内容、Web 事件或持久化。

## 调用路径

- FastAPI 单论文接口只调用 `PaperAnalysisOrchestrator` 并序列化事件。
- Orchestrator 负责解析、视觉检查、证据、持久化和完整任务状态。
- LangGraph 负责三个专业 Agent 的并行依赖、Summary fan-in 与 Assessment；
  实际 Agent 执行统一委托给 Harness。
- 原有 `run_*_agent` / `stream_*_agent` 函数保留为兼容入口，但其内部同样调用
  Harness，不再直接调用模型。
- 多论文 Comparison 的 token 流继续保持现有前端协议，执行本身已接入 Harness。

## 扩展新 Agent

新增 Agent 时应：

1. 定义输出 Pydantic schema 和 Prompt 构造函数；
2. 在 Agent 模块声明 `AgentSpec`；
3. 由 Web 或 LangGraph 使用 `get_agent_harness().run(...)`；
4. 为检索、事件、结构校验和错误路径补充测试；
5. 不在编排代码中重复实现重试、进度缓冲或 provider 判断。

## 第一版边界

- 继续复用现有 `utils.llm` 厂商适配，不改变模型配置与认证方式；
- 保持既有 Web 流式事件及前端展示契约；
- 不引入新的队列、分布式调度器或外部数据库；
- Runtime 元数据目前用于单次执行诊断，尚未新增独立持久化表。
