# Multi-Agent Paper Reader

[English](./README.md) | **简体中文**

Multi-Agent Paper Reader 是一个基于证据的学术论文研读助手。用户上传 PDF 后，系统会解析论文结构，根据正文、提取的表格以及视觉模型生成的图像摘要建立可追溯证据索引，再由多个专职 Agent 分别完成方法分析、实验分析和批判性评审，最后生成结构清晰的论文研读笔记。

![Paper Reader 论文研读工作台](./docs/assets/paper-reader-workspace.png)

## Web 应用

本仓库包含一个完整的全栈 Web 应用：

- 后端：基于 FastAPI 的 `app.py`
- 前端：基于 React + Vite 的 `frontend-prototype/`
- 分析 API：`POST /api/analyze` 接收 PDF 文件并返回论文元数据及全部 Agent 输出
- 流式分析 API：`POST /api/analyze/stream` 以换行分隔的 JSON 事件返回解析进度、证据索引、模型 Token、Agent 完成状态和最终总结
- 论文追问 API：`POST /api/chat/stream` 综合近期对话、精简长期记忆索引、与问题相关的主题记忆、召回的早期消息以及论文原文证据，流式生成回答
- 会话 API：`GET/POST /api/history/{id}/conversations` 和 `GET/PATCH/DELETE /api/chat/conversations/{id}` 支持为同一篇论文创建、恢复、重命名和删除多个持久化会话
- 多论文对比 API：`POST /api/comparisons/stream` 对 2～4 篇历史论文执行带证据前缀的比较，`/api/comparisons/*` 用于持久化对比工作区及跨论文会话
- 历史 API：`GET /api/history`、`GET /api/history/{id}` 和 `DELETE /api/history/{id}` 用于保存、恢复和删除已完成的论文分析
- 设置 API：`GET /api/settings` 在不暴露凭据的前提下返回厂商目录、兼容协议与当前路由；保存凭据时会通过所选协议和文本模型发起最小真实请求，验证成功后才写入本机
- 章节标题：常见标题使用本地中文词典转换；无法识别的英文标题会在正式分析开始前由当前文本模型通过一次有界批量请求完成翻译
- 静态托管：FastAPI 从 `frontend-prototype/dist` 提供构建后的 React 应用

本地运行：

```bash
# Python 后端依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 前端依赖与生产构建
cd frontend-prototype
npm install
npm run build
cd ..

# 启动全栈应用
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/
```

首次使用时，可直接打开右上角的 **Settings**。内置路由支持 GLM、DeepSeek、OpenAI、Qwen、Doubao、Anthropic 和 Kimi。模型目录以厂商官方 API 文档中的真实 ID 为准：包括 Claude Fable 5 / Sonnet 5 / Opus 4.8、Kimi K2.6、Doubao Seed 2.1 Pro / Turbo、GPT-5.6 Sol、Qwen3.7 与 DeepSeek V4 等当前主线模型。Anthropic 使用原生 Messages API；其余兼容厂商使用 OpenAI Chat Completions 协议。GLM-5.2 提供“标准思考 / 深度思考 / 快速响应”，其中深度模式会真实发送 `reasoning_effort=max`；Qwen3.7/3.6 混合思考模型通过 `enable_thinking` 在“深度思考 / 快速响应”间切换，深度模式不人为限制 `thinking_budget`，使用厂商规定的模型默认上限；Kimi K2.6 与 DeepSeek V4 则发送真实 `thinking.type`。视觉理解始终跟随文本服务商：所选服务有明确视觉模型时可以启用，没有视觉模型时则明确关闭。

Settings 还提供“自定义中转站”。用户必须明确选择 `OpenAI-compatible` 或 `Anthropic-compatible`，并填写 Base URL、文本模型 ID 与可选视觉模型 ID。视觉模型 ID 留空就表示该中转站不启用图表理解。保存时后端会按所选协议使用文本模型发起最小真实请求；只有请求成功后才将 API Key 和中转配置写入本机 `.env`，且任何已保存 Key 都不会回传浏览器。由于请求内容会发送给所配置的第三方服务，使用中转站前应自行确认其隐私、计费和可靠性。

默认路由仍为智谱 `glm-5.2` 文本模型，并自动配对 `glm-5v-turbo` 视觉模型。也可以将 `.env.example` 复制为 `.env`，手动配置 `TEXT_PROVIDER`、`MODEL_NAME` 以及相应厂商的 Key。为兼容旧配置，`VISION_PROVIDER` 仍会被读取，但运行时会强制归一为 `TEXT_PROVIDER`；内置厂商使用目录中的推荐视觉模型，自定义中转站使用用户明确填写的视觉模型 ID。Agent 生成温度由 `LLM_TEMPERATURE` 控制；基于证据的论文追问使用独立的低温配置 `CHAT_TEMPERATURE`，默认值为 `0.25`。`CHAT_INPUT_TOKEN_BUDGET` 用于设置证据、近期对话和长期记忆共享的保守动态输入预算，默认值为 `48000`。

如需理解论文中的图像和图表，请设置 `ENABLE_VISION_SUMMARY=true`，并选择带有官方托管视觉模型的文本厂商。后端会将 PDF 中的视觉区域渲染为 PNG，默认并发请求各个选中的图像或图表，让自动配对的视觉模型生成简洁中文摘要，并将其记录为 `F` 类证据。如果供应商返回限流错误，失败图像会自动使用更小的 `VISION_RETRY_WORKERS` 并发池重试。

## CLI 快速开始

```bash
pip install -r requirements.txt
copy .env.example .env
# 编辑 .env 并填写当前所选厂商的 API Key

python main.py examples/your_paper.pdf --pretty
```

## 验证

每次推送 `main` 后，`.github/workflows/ci.yml` 都会在 macOS、Windows 和 Ubuntu 上运行完整后端测试及 Vite 生产构建。发布前如需进行带凭据的真实检查，`tools/provider_smoke.py` 会分别执行一次最小真实请求和一次要求引用证据的论文追问，并且只输出非敏感调用溯源字段：

```bash
python tools/provider_smoke.py openai deepseek doubao
```

在仓库配置 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY` 和 `ARK_API_KEY` Secrets 后，也可以手动触发 `Live provider smoke tests` 工作流运行同一检查。

## 系统架构

```text
PDF
-> core.pdf_parser.parse_pdf
-> core.evidence.build_evidence_index
-> MethodAgent + ExperimentAgent + CriticAgent 读取相关正文/表格/图像证据片段
-> SummaryAgent 综合各 Agent 输出及其证据
-> 结构化论文研读笔记
```

实时分析会发送以下事件类型：

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

`agent_token` 是模型实时生成的原始 Token。前端会先将其显示为实时预览；JSON 对象生成完成后，再展示经过 Pydantic 解析的结构化结果。

## 多论文对比

通过 `Reading Workspace` 菜单切换到 `Comparison Workspace`，即可选择 2～4 篇已经完成分析的论文。ComparisonAgent 会复用每篇论文的 Method、Experiment、Critic 和 Summary 结构化结果，同时根据均衡选取的论文原文证据重新核验各项判断。证据 ID 会增加论文前缀，例如 `P1:E003`、`P2:T001` 或 `P3:F002`，避免不同 PDF 中重复出现的 `E001` 被混为同一来源。

对比结果包括可横向滚动的完整矩阵、可直接比较/需结合条件/不宜直接比较状态、数据集与指标不一致警告、可点击证据预览、跨论文研究空白、条件化适用建议以及由后端确定性计算的证据覆盖率。完成的对比任务、论文关联和跨论文追问会话都会保存到同一 SQLite 数据库，刷新浏览器或重启后端后仍可恢复。跨论文追问会从每篇论文中均衡检索相关证据，而不是将多个 PDF 全文一次性发送给模型。

## 可解释评估

每个完成的 API 响应都包含一个 `assessment` 对象，其中包括两个相互独立的结果：

- **创新性评分（1-5）：** Critic Agent 分别评估问题原创性（15%）、方法原创性（40%）、与已有工作的差异（30%）以及方法通用性（15%）。后端计算加权总分，并保留每一项评分理由及支持该判断的证据 ID。
- **分析可靠度（0-100）：** 后端根据 PDF 解析质量（20 分）、关键章节覆盖度（35 分）、有效证据引用（30 分）以及结构化输出完整性（15 分）进行确定性计算。

分析可靠度不是模型自行报告的“信心”。当相关工作覆盖不足、有效引用少于三条、解析内容不足、创新性维度不完整或处于 Demo 模式时，系统会应用明确的分数上限。响应中会公开各项得分、原始分数、限制上限、最终分数和警告，方便用户审计结果。

## 论文追问

用户可以通过结果面板右下角的 AI 按钮直接打开论文追问，也可以选中一段分析结果后点击 **在侧边聊天中提问**，将该片段作为当前问题的引用。每篇论文可以建立多个独立会话，并通过追问面板顶部的下拉框切换、重命名或删除。所有用户原始问题和模型回答都会保存到 SQLite，在刷新浏览器或重启后端后仍可恢复。

首条问题发送后，系统会立即给出精简的本地临时标题，再由后台 GLM 将其整理为简短的主题摘要；如果用户已经手动改名，自动标题不会覆盖该名称。追问内容支持 GFM 表格以及由 KaTeX 渲染的行内和块级数学公式。流式回答只会在用户停留于底部时自动跟随新 Token；用户主动向上阅读后会暂停跟随，并显示一键回到最新回答的按钮。

长对话架构参考了 Claude Code 公开的记忆模式：每轮自动加载一份精简记忆索引，仅在问题相关时召回详细主题记忆。最近六个完整问答轮次会保留原文；每当又有六个较早轮次满足压缩条件时，后台 GLM 任务会将它们整理进记忆索引和主题记录，同时保留全部原始消息，也不会阻塞当前回答。每个问题还可以按需召回相关的早期原始消息。动态 Token 预算会优先分配给当前问题、选中片段、论文原文证据、近期对话、分析上下文、记忆和外部资料，不再依赖固定消息条数。

每次完成正式分析后，系统都会返回一个不透明的 `analysis_id`。后端在有界的四小时内存缓存中保留该分析对应的完整 `E`/`T`/`F` 证据片段。收到问题后，系统会综合中英文查询词、对话上下文、Agent 引用的证据 ID 和章节意图，从论文原文中选出最相关的片段。回答 Prompt 将论文原文证据视为最高依据，要求标注证据 ID 和页码，并明确区分论文事实、背景知识、长期记忆与模型推断。

当用户明确询问近期工作、相关论文或与其他论文的对比时，系统还可以使用 Semantic Scholar 提供的题录和摘要信息。该查询是可选能力，失败时会安全降级；可以设置 `SEMANTIC_SCHOLAR_API_KEY` 以使用独立 API 配额。Sample 和 Demo 结果采用确定性回复，因此无需额外调用模型也能验证完整交互流程。重新打开已保存论文时，系统会从持久化的完整证据中重新建立实时聊天证据会话。

## 论文历史

每次完成的上传默认保存在本地 `.paper-reader/` 目录。SQLite 数据库中包含论文元数据、结构化 Agent 输出、评估结果、完整证据片段、单篇与多论文对比工作区、聊天会话、不可变原始消息、记忆索引和主题记忆；原始 PDF 保存在 `.paper-reader/papers/`。再次上传同一份 PDF 时，系统会更新已有历史记录，而不是重复创建。`Recent Papers` 和顶部 History 菜单均从该数据库读取数据，因此用户刷新浏览器或重启后端后，可以直接恢复论文分析及其全部会话，无需重新上传 PDF。重新打开正式分析结果时，系统还会根据已保存证据重建追问所需的证据会话。

可以通过 `PAPER_READER_DATA_DIR` 修改全部历史数据的存储位置，或通过 `PAPER_HISTORY_DB` 指定 SQLite 文件路径。从 History 菜单删除论文时，其数据库记录、追问会话和保留的 PDF 文件都会一并删除。

原始架构说明请参阅 [CLAUDE.md](./CLAUDE.md)。

## 技术栈

- LangGraph：编排多 Agent 工作流
- PyMuPDF：PDF 解析、基于目录的章节识别、表格提取和视觉区域渲染
- Pydantic v2：结构化输出 Schema 与结果校验
- 证据片段：为结论提供页码和章节依据（`E` 正文、`T` 表格、`F` 视觉图像摘要）
- FastAPI：后端 API 与静态文件托管
- React + Vite：前端论文研读工作台
