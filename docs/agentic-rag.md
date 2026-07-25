# Agentic RAG 架构

## 目标

Agentic RAG 让所有文本模型路由都具备“先判断证据是否够用，再按需调用论文工具”
的能力，而不是只接收后端预先挑好的固定片段。Codex 仍保留本机 MCP 工具上下文；
OpenAI、Anthropic、GLM、DeepSeek、Qwen、Doubao、Kimi 和自定义中转站通过同一
Harness 接入，不在业务流程中写死厂商。

```text
AgentSpec
  -> deterministic seed evidence
  -> AgenticRetrievalRuntime
       |- native tool call (when declared and accepted)
       `- structured RetrievalDecision fallback (every provider)
  -> PaperToolRegistry executes one read-only action
  -> observation returned to retrieval controller
  -> finish / next action / hard budget stop
  -> final structured AgentRuntime call
```

## 工具

`core/paper_tools.py` 提供能力受限的通用工具：

- `paper_search`：对完整证据索引执行新的语义 + 词法混合检索；
- `paper_overview`：查看标题、章节、页码范围和图表数量；
- `paper_read_section` / `paper_read_page`：按章节或一页读取证据；
- `paper_read_table` / `paper_read_figure`：读取已索引的 T/F 证据；
- `calculate`：只允许 AST 白名单内的小型算术；
- `finish_retrieval`：模型确认当前证据足够后结束循环。

这些工具不接受文件路径、URL、任意 SQL、shell、Python 代码或模型提供的坐标。
论文正文、图表文本、用户问题和上游 Agent 输出都被视为数据，不具备系统指令权限。

## 双策略 Runtime

`core/model_providers.py` 中的能力注册表只决定是否尝试原生工具调用。原生调用是
性能优化，不是正确性的前提：

1. `AGENTIC_TOOL_STRATEGY=auto` 时，声明支持的路由先尝试 `bind_tools`；
2. 模型、网关或具体版本拒绝工具调用、没有返回工具动作或参数不合法时，立即切换到
   `RetrievalDecision` 结构化动作；
3. 结构化动作仍失败时，保留原来的确定性 seed evidence，继续最终 Agent 调用；
4. 不会在不同模型厂商之间静默改道。

`RAG_MODE=hybrid` 可完整关闭模型驱动循环，恢复原来的一次确定性混合检索。该开关
用于回滚和 A/B，不会改变证据索引或历史数据格式。

## 预算和停止条件

默认每个 Agent 最多执行 4 个动作，单步最多返回 6 个片段，最终最多保留 16 个、
24,000 字符的证据。重复动作会立即停止，非法工具参数会作为安全观察返回给模型，
不会执行越界操作。达到预算后，系统使用当前证据继续，并要求最终输出保留不确定性。

相关环境变量：

```text
RAG_MODE=agentic|hybrid
AGENTIC_TOOL_STRATEGY=auto|native|structured
AGENTIC_RAG_MAX_STEPS
AGENTIC_RAG_MAX_RESULTS_PER_STEP
AGENTIC_RAG_MAX_EVIDENCE_ITEMS
AGENTIC_RAG_MAX_EVIDENCE_CHARS
AGENTIC_RAG_MAX_OBSERVATION_CHARS
AGENTIC_RAG_PLANNER_RETRIES
AGENTIC_RAG_CHECKPOINTS
```

## 专业 Agent 与证据监督

Method、Experiment、Critic 各自拥有不同的检索目标，但不再硬隔离证据类型：
Experiment 的目标会提示优先核对含精确数值的表格，Method 会在架构图确有帮助时
读取图像证据，最终选择仍由当前任务、已返回证据和模型下一步动作共同决定。

三个专业 Agent 完成后，`EvidenceSupervisor` 同时执行：

- 模型语义检查：结论与引用内容是否真正对应、是否有关键缺口或冲突；
- 后端确定性检查：引用 ID 是否存在于当前论文证据索引；
- 最多一次 repair：只重新运行被点名的专业 Agent，并把缺失 facet 和建议查询传入
  新一轮检索；修复后不再循环。

Supervisor 模型不可用时，确定性检查继续工作，不会让已经完成的整篇分析失效。
Summary 会获得上游关键结论、有效引用 seed 和一次按需补查机会。

## 论文追问与多论文对比

单论文追问、对比生成和跨论文追问复用相同的 `AgenticRetrievalRuntime`。
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

`tools/benchmark_agentic_rag.py` 对同一 source-anchored gold set 比较：

- A0：词法/静态路由；
- A1：确定性混合检索；
- A2：结构化动作 Agentic RAG；
- A3：原生工具 Agentic RAG；
- A4：Agentic RAG + Evidence Supervisor/repair。

指标包含 Recall@10、Precision@10、MRR、nDCG@10、Grounded Fact F1、工具成功率、
调用次数和延迟。每个 gold item 必须同时保存 PDF SHA-256 与稳定的页码/引文哈希
source anchor，不能把一次运行中的 `E003` 单独当成长期 gold。测试中的合成用例只
验证评分器，不代表整个系统的实际准确率。
