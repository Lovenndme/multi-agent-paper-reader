# Agentic RAG 架构

## 目标

Agentic RAG 让所有文本模型路由都具备“先使用本地语义证据，再按缺口调用论文工具”
的能力，而不是让每个 Agent 都无条件增加一次模型规划。Codex 仍保留本机 MCP 工具上下文；
OpenAI、Anthropic、GLM、DeepSeek、Qwen、Doubao、Kimi 和自定义中转站通过同一
Harness 接入，不在业务流程中写死厂商。

```text
AgentSpec
  -> deterministic hierarchical seed evidence
  -> AgenticRetrievalRuntime
       |- global mode: hybrid / adaptive / agentic
       |- request policy: skip / auto / force
       |- native tool call (when declared and accepted)
       `- structured RetrievalDecision fallback (every provider)
  -> PaperToolRegistry executes one read-only action
  -> observation returned to retrieval controller
  -> finish / next action / hard budget stop
  -> final structured AgentRuntime call
```

## 确定性检索主链路

`core/hybrid_retrieval.py` 是专业 Agent、论文追问、论文工具和多论文对比共享的
检索核心。PDF 展示文本保持原样用于引用；检索副本单独清理软连字符、断行连字符和
多余空白，避免 `pa-\nrameter` 等解析噪声破坏匹配。每条父证据再按
`cl100k_base` 切成 220-token、40-token overlap 的子窗口：

```text
EvidenceSnippet parent
  -> retrieval-only PDF normalization
  -> token-aware subchunks
  -> Dense 子块召回，按最大得分回并到 parent
  -> parent 级 BM25
  -> Dense/BM25 Reciprocal Rank Fusion
  -> Top-24 候选 Cross-Encoder 重排
  -> parent 证据去重、章节配额和字符预算
```

子块只用于定位，最终返回完整 `EvidenceSnippet`，因此证据 ID、页码、图表类型和引用
锚点不会在切片后丢失。Embedding、Cross-Encoder 或 tokenizer 不可用时分别退化为
BM25、保留融合排序和确定性字符窗口，不会中断论文分析。职责 seed 为避免三个专业
Agent 首轮都支付重排成本，使用相同的分层召回但关闭 Cross-Encoder；用户问题、
论文工具检索和多论文查询默认启用重排。默认 MiniLM Cross-Encoder 只用于英文
query；中文 query 保留多语言 Dense/BM25 融合结果，避免用英文单语模型制造虚假
高置信度，也可显式配置经验证的多语言 reranker。

## 工具

`core/paper_tools.py` 提供能力受限的通用工具：

- `paper_search`：对完整证据索引执行分层 BM25 + Dense RRF + Cross-Encoder 检索；
- `paper_overview`：查看标题、章节、页码范围和图表数量；
- `paper_read_section` / `paper_read_page`：按章节或一页读取证据；
- `paper_read_table` / `paper_read_figure`：读取已索引的 T/F 证据；
- `calculate`：只允许 AST 白名单内的小型算术；
- `finish_retrieval`：模型确认当前证据足够后结束循环。

这些工具不接受文件路径、URL、任意 SQL、shell、Python 代码或模型提供的坐标。
论文正文、图表文本、用户问题和上游 Agent 输出都被视为数据，不具备系统指令权限。

## 模式、请求策略与双策略 Runtime

`core/model_providers.py` 中的能力注册表只决定是否尝试原生工具调用。原生调用是
性能优化，不是正确性的前提：

1. `AGENTIC_TOOL_STRATEGY=auto` 时，声明支持的路由先尝试 `bind_tools`；
2. 模型、网关或具体版本拒绝工具调用、没有返回工具动作或参数不合法时，立即切换到
   `RetrievalDecision` 结构化动作；
3. 结构化动作仍失败时，保留原来的确定性 seed evidence，继续最终 Agent 调用；
4. 不会在不同模型厂商之间静默改道。

全局模式与请求策略相互独立：

- `hybrid`：始终只使用确定性语义检索，用于回滚；
- `adaptive`：默认模式，仅在请求策略为 `force`，或 `auto` 门控识别到数值、图表、
  跨章节、低覆盖等证据风险时调用 planner；
- `agentic`：忽略请求级跳过策略，始终运行完整规划循环，用于 A/B 基线；
- `skip / auto / force`：分别用于首轮专业 Agent、论文追问和 Supervisor 定向修复。

优先级固定为 `hybrid > agentic > adaptive policy`，因此测试基线不会被业务门控污染。
`AgenticRagConfig`、模型路由和 `AgenticRunBudget` 会在分析或追问请求开始时冻结，
并沿 `AgentRunContext` 传递；并行 Agent 不通过临时改写环境变量切换策略。

## 预算和停止条件

全量 `agentic` 基线默认每个 Agent 最多执行 4 个动作；`adaptive` 的定向检索默认
最多 2 步，Summary 最多 1 步。单步最多返回 6 个片段，最终最多保留 16 个、
24,000 字符的证据。重复动作会立即停止，非法工具参数会作为安全观察返回给模型，
不会执行越界操作。达到预算后，系统使用当前证据继续，并要求最终输出保留不确定性。
字符上限是硬约束：单条超长表格或图像摘要会被截断；多论文输入按来源轮询并为尚未
出现的论文预留字符份额，避免第一篇论文耗尽整个上下文。
可选的 `AGENTIC_RAG_PLANNER_MODEL/MODE` 只改变同一厂商内的检索规划路由，不改变
最终专业 Agent 使用的模型；非法覆盖会安全回落到本次请求冻结的最终模型路由。

相关环境变量：

```text
RAG_MODE=adaptive|agentic|hybrid
AGENTIC_TOOL_STRATEGY=auto|native|structured
AGENTIC_RAG_MAX_STEPS
AGENTIC_RAG_ADAPTIVE_MAX_STEPS
AGENTIC_RAG_ADAPTIVE_SUMMARY_MAX_STEPS
AGENTIC_RAG_ADAPTIVE_MIN_SEED_ITEMS
AGENTIC_RAG_ADAPTIVE_MIN_SEED_CHARS
AGENTIC_RAG_PLANNER_MODEL
AGENTIC_RAG_PLANNER_MODE
AGENTIC_RAG_MAX_RESULTS_PER_STEP
AGENTIC_RAG_MAX_EVIDENCE_ITEMS
AGENTIC_RAG_MAX_EVIDENCE_CHARS
AGENTIC_RAG_MAX_OBSERVATION_CHARS
AGENTIC_RAG_PLANNER_RETRIES
AGENTIC_RAG_CHECKPOINTS
EMBEDDING_MODEL
PAPER_READER_EMBEDDING_BATCH_SIZE
PAPER_READER_RETRIEVAL_SUBCHUNK_TOKENS
PAPER_READER_RETRIEVAL_SUBCHUNK_OVERLAP
PAPER_READER_RETRIEVAL_CANDIDATES
PAPER_READER_RERANKER_ENABLED
PAPER_READER_RERANKER_MODEL
```

## 专业 Agent 与证据监督

Method、Experiment、Critic 首轮各自使用职责感知的 FastEmbed seed，不增加 planner
调用；它们拥有不同的检索目标，但不硬隔离证据类型：
Experiment 的目标会提示优先核对含精确数值的表格，Method 会在架构图确有帮助时
读取图像证据，最终选择仍由当前任务、已返回证据和模型下一步动作共同决定。

三个专业 Agent 完成后，`EvidenceSupervisor` 同时执行：

- 模型语义检查：结论与引用内容是否真正对应、是否有关键缺口或冲突；
- 后端确定性检查：引用 ID 是否存在于当前论文证据索引；
- 最多一次 repair：只重新运行被点名的专业 Agent，并以 `force` 策略把缺失 facet
  和建议查询传入最多两步的新一轮检索；修复后不再循环。

Supervisor 模型不可用时，确定性检查继续工作，不会让已经完成的整篇分析失效。
Summary 会获得上游关键结论和有效引用 seed；仅当 Supervisor 修复后仍判定不足或
保留 warning 时，才获得最多一次补查机会。`coverage_score` 明确定义为 0–100
百分制，但自适应门控使用 `sufficient / repair_tasks / warnings`，不依赖单一分数。

## 论文追问与多论文对比

单论文追问、对比生成和跨论文追问复用相同的 `AgenticRetrievalRuntime`。简单定义、
概括或已命中 seed 的追问跳过 planner；精确数值、表格/图像/公式、跨章节比较、
证据冲突或明显低覆盖的问题触发最多两步检索。多论文对比生成固定为 `force`，
跨论文追问仍由确定性问题门控决定。
中文问题不会仅因分词覆盖偏低而一律触发：短定义题保持静态，而有足够实质词项且
覆盖接近零的问题会补检索；显式引用某个证据 ID 也不能绕过精确数值核验。
多论文证据 ID 在工具层加上 P1/P2 前缀，检索结果按论文分组回填，防止事实串台。
最终回答仍由原有流式文本路径生成，Agentic 阶段只负责证据选择。

## 公开事件与持久化

分析和追问流可以发送：

```text
retrieval_started -> query_planned -> tool_started -> tool_complete
-> evidence_selected -> evidence_graded -> query_refined
-> coverage_checked -> repair_started -> retrieval_complete
```

事件只包含公开过程摘要、工具名、计数和耗时；不包含隐藏思维链、原始结构化 JSON、
模型 prompt、API Key 或论文全文。前端在生成期间展示这些摘要，最终结果完成后与
其他过程记录一起折叠。

完整分析结束时，公开 trace 会随历史结果保存在本机。可选的
`.paper-reader/agentic-rag.sqlite3` 仅追加公开工具循环 checkpoint，便于诊断中断
位置；它不保存模型 prompt 或隐藏推理，也不宣称可以恢复未完成的模型调用。

## 评测

评测分为 source-anchored 检索与完整主链路两层：

- A0：词法职责静态检索；
- A1：FastEmbed 职责静态检索；
- AQL：改造前的父块 query-aware 检索；
- AQ：token 子块 + parent 回并 + BM25/Dense RRF；
- AQR：AQ + 本地 Cross-Encoder 重排；
- A2：结构化动作 Agentic RAG；
- A3：原生工具 Agentic RAG；
- A4：自适应原生工具 Agentic RAG；
- B0/B1/B2：在相同专业 Agent、Evidence Supervisor/repair 和 Summary 下，对比
  Hybrid、全量 Native Agentic 与 Adaptive 完整链路。

`tools/run_agentic_rag_benchmark.py` 执行 A0–A4，`tools/benchmark_agentic_rag.py`
合并和评分一次或多次运行。主检索指标为前排质量更严格的 Recall@5，并用
Recall@10/16 检查生产证据预算内的覆盖；同时报告 Precision、MRR、nDCG、工具
成功率、调用次数、回退、错误和延迟。每个 Gold
必须保存 PDF SHA-256 与稳定的页码/引文哈希 source anchor，不能把一次运行中的
`E003` 单独当成长期 Gold。

完整主链路在没有人工事实 Gold 时只报告 Schema、引用有效性、Supervisor/repair 和
端到端性能，不把内部评分包装为论文解读准确率。协议、Pilot 结果和正式集扩展要求见
`docs/agentic-rag-benchmark.md`。测试中的合成用例只验证评分器，不代表整个系统的
实际准确率。
