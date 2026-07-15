# Multi-Agent Paper Reader

[English](./README.md) | **简体中文**

Multi-Agent Paper Reader 是一个基于证据的学术论文研读助手。用户上传 PDF 后，系统会解析论文结构，根据正文、提取的表格以及视觉模型生成的图像摘要建立可追溯证据索引，再由多个专职 Agent 分别完成方法分析、实验分析和批判性评审，最后生成结构清晰的论文研读笔记。

![Paper Reader 论文研读工作台](./docs/assets/paper-reader-workspace.png)

## Web 应用

本仓库包含一个完整的全栈 Web 应用：

- 后端：基于 FastAPI 的 `app.py`
- 前端：基于 React + Vite 的 `frontend-prototype/`
- 分析 API：`POST /api/analyze` 接收 PDF 文件并返回论文元数据及全部 Agent 输出
- 预解析 API：`POST /api/papers/preview` 在文件选择后立即返回论文标题、页数、章节数和原始章节目录，不调用模型或写入历史
- 流式分析 API：`POST /api/analyze/stream` 以换行分隔的 JSON 事件返回解析进度、证据索引、模型 Token、Agent 完成状态和最终总结
- 论文追问 API：`POST /api/chat/stream` 综合近期对话、精简长期记忆索引、与问题相关的主题记忆、召回的早期消息以及论文原文证据，流式生成回答
- 会话 API：`GET/POST /api/history/{id}/conversations` 和 `GET/PATCH/DELETE /api/chat/conversations/{id}` 支持为同一篇论文创建、恢复、重命名和删除多个持久化会话
- 多论文对比 API：`POST /api/comparisons/stream` 对 2～4 篇历史论文执行带证据前缀的比较，`/api/comparisons/*` 用于持久化对比工作区及跨论文会话
- 历史 API：`GET /api/history`、`GET /api/history/{id}` 和 `DELETE /api/history/{id}` 用于保存、恢复和删除已完成的论文分析
- 设置 API：`GET /api/settings` 在不暴露凭据的前提下返回厂商目录、兼容协议与当前路由；`/api/settings/codex/*` 返回本机 Codex 登录/模型状态并启动仅限 localhost 的 ChatGPT 登录流程；API Key 通过最小真实请求验证成功后才写入本机
- 章节标题：目录保留 PDF 解析得到的原始标题语言，仅压缩异常空白，并在标题明显损坏时使用编号占位
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

首次使用时，可直接打开右上角的 **Settings**。内置路由支持 GLM、DeepSeek、OpenAI、Qwen、Doubao、Anthropic、Kimi 和 Codex 订阅。API 厂商模型目录以官方文档中的真实 ID 为准：包括 Claude Fable 5 / Sonnet 5 / Opus 4.8、Kimi K2.6、Doubao Seed 2.1 Pro / Turbo、GPT-5.6 Sol、Qwen3.7 与 DeepSeek V4 等当前主线模型。Anthropic 使用原生 Messages API；其余兼容厂商使用 OpenAI Chat Completions 协议。GLM-5.2 提供“标准思考 / 深度思考 / 快速响应”，其中深度模式会真实发送 `reasoning_effort=max`；Qwen3.7/3.6 混合思考模型通过 `enable_thinking` 在“深度思考 / 快速响应”间切换，深度模式不人为限制 `thinking_budget`，使用厂商规定的模型默认上限；Kimi K2.6 与 DeepSeek V4 则发送真实 `thinking.type`。视觉理解始终跟随文本服务商：所选服务有明确视觉模型时可以启用，没有视觉模型时则明确关闭。

### Codex 订阅路由（仅限本机单用户）

可选的 Codex 路由使用 `requirements.txt` 中按不可变源码归档锁定的官方 `openai-codex` Python SDK，并固定配套 runtime `0.144.4`。该源码版本能够识别当前 GPT-5.6 目录中的 `max` 和 `ultra`；路由启用前还会核对 SDK 来源 revision、归档 SHA-256、runtime 包和实际二进制版本。要求 Python 3.10 或更高版本，但不要求额外安装全局 Codex CLI。可参考官方 [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk)、[模型说明](https://learn.chatgpt.com/docs/models)、[认证说明](https://learn.chatgpt.com/docs/auth)及本项目固定的 [SDK 源码 revision](https://github.com/openai/codex/commit/3f74f00295dcb1346340686bb09c5bfd4f0237c4)。

打开 **Settings → Codex 订阅** 后，可选择官方浏览器登录或设备码登录。本机已有 Codex/ChatGPT 会话时会直接复用；否则由官方 runtime 完成登录，并按 Codex 原有机制保存在本机认证缓存。网页不会接收、读取、保存或返回 Codex token。主动“断开连接”会退出这份本机共享会话，因此界面会提示其他本机 Codex 客户端可能也需要重新登录。

只有账号实时 `model/list` 返回才是路由依据。登录账号真实返回时，项目仅保留 `gpt-5.6-sol`、`gpt-5.6-terra` 和 `gpt-5.6-luna`；目录失败、为空或不包含某模型时，不会静默回退到过期/静态模型。Settings 始终展示 `low`、`medium`、`high`、`xhigh`、`max`、`ultra` 六个推理位置，并禁用所选模型没有返回的档位。当前实测目录中 Sol、Terra 支持六档，Luna 支持前五档并明确禁用 Ultra。

Codex 可用于专职 Agent、结构化总结、单篇/多篇论文追问、会话标题、图表摘要及 JSON Schema 记忆更新。每次调用都使用隔离的临时线程、拒绝审批、只读沙箱，并关闭 Codex 历史持久化。所选模型和固定版本 runtime 支持时，原生 Web Search、规划、工具发现、图像查看与图像生成均予以保留；联网来源会与论文原文证据分开展示，图像生成只在用户明确提出时使用。绑定单篇论文的调用还会获得八个宿主提供的只读论文工具：论文证据检索、章节/页读取、图/表元数据、仅限解析器已验证区域的视觉裁剪、当前论文记忆召回和受限计算器。集成会禁用继承的 MCP Server、Apps、Plugins、Hooks、Memories、技能注入、安装建议、交互式提问、Shell、统一执行与任意文件写入，并在启动 SDK 前检查最终 MCP 目录，隔离失败即拒绝启动。生成图片是唯一受控写入例外，由官方 runtime 保存到本机 `CODEX_HOME/generated_images` 目录。Ultra 最多创建两个一层子 Agent；其余五档的线程容量固定为 1，因此即使模型元数据声明了协作工具也无法真正创建子线程。安全边界参见官方[审批与沙箱说明](https://learn.chatgpt.com/docs/agent-approvals-security)和 [MCP 文档](https://learn.chatgpt.com/docs/extend/mcp)。

该路由只适用于通过 loopback 地址访问的可信本机单用户应用，不是公网多用户认证方案，不能让远程访客消耗机器所有者的 ChatGPT 订阅。登录/退出接口同时校验 loopback 客户端、loopback Host 和浏览器同源请求，以防跨站调用与 DNS 重绑定。若要公开或多人部署，应使用常规厂商 API 凭据，并自行实现用户认证、额度与计费。

完整安装步骤、路由规则、工具矩阵和升级清单见 [docs/codex-subscription.md](./docs/codex-subscription.md)。

Settings 还提供“自定义中转站”。用户必须明确选择 `OpenAI-compatible` 或 `Anthropic-compatible`，并填写 Base URL、文本模型 ID 与可选视觉模型 ID。视觉模型 ID 留空就表示该中转站不启用图表理解。保存时后端会按所选协议使用文本模型发起最小真实请求；只有请求成功后才将 API Key 和中转配置写入本机 `.env`，且任何已保存 Key 都不会回传浏览器。由于请求内容会发送给所配置的第三方服务，使用中转站前应自行确认其隐私、计费和可靠性。

默认路由仍为智谱 `glm-5.2` 文本模型，并自动配对 `glm-5v-turbo` 视觉模型。也可以将 `.env.example` 复制为 `.env`，手动配置 `TEXT_PROVIDER`、`MODEL_NAME` 以及相应厂商的 Key。为兼容旧配置，`VISION_PROVIDER` 仍会被读取，但运行时会强制归一为 `TEXT_PROVIDER`；内置厂商使用目录中的推荐视觉模型，自定义中转站使用用户明确填写的视觉模型 ID。Agent 生成温度由 `LLM_TEMPERATURE` 控制；基于证据的论文追问使用独立的低温配置 `CHAT_TEMPERATURE`，默认值为 `0.25`。`CHAT_INPUT_TOKEN_BUDGET` 用于设置证据、近期对话和长期记忆共享的保守动态输入预算，默认值为 `48000`。

如需理解论文中的图像和图表，请设置 `ENABLE_VISION_SUMMARY=true`，并选择带有官方托管视觉模型的文本厂商。后端使用 PyMuPDF4LLM Layout 区分正文、公式、表格、图注和 picture 区域；picture 同时覆盖嵌入位图与 PDF 矢量图。图注通过同页几何位置与 picture 配对，不再按两个无关列表的序号硬配；视觉模型只接收具有已验证 bbox 的精确裁剪，找不到区域时会跳过，绝不会静默退化成整页截图。失败图像会自动使用更小的 `VISION_RETRY_WORKERS` 并发池重试。

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
-> PyMuPDF4LLM Layout 分类正文、公式、表格、图注、位图与矢量图
-> core.pdf_parser.parse_pdf 构建原始语言章节和精确视觉区域
-> core.evidence.build_evidence_index + 本地多语言 Embedding 语义排序
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

输入框底部可以从已经验证并配置 Key 的厂商或已连接的本机 Codex 订阅中，为当前问题选择文本模型及其响应模式。该选择只影响本次追问，不会改写 Settings 中的全局路由。选择器和回答底部只显示一次模型名称，不显示确认文字，也不会展开厂商端点、上游响应模型或 Request ID。每条模型回答均可一键复制；论文证据 ID 和页码继续用于后端检索与回答约束，但不会显示在最终回答中。

首条问题发送后，系统会立即给出精简的本地临时标题，再由后台 GLM 将其整理为简短的主题摘要；如果用户已经手动改名，自动标题不会覆盖该名称。追问内容支持 GFM 表格以及由 KaTeX 渲染的行内和块级数学公式。流式回答只会在用户停留于底部时自动跟随新 Token；用户主动向上阅读后会暂停跟随，并显示一键回到最新回答的按钮。

记忆层现已统一使用 LangMem 0.0.30，不再保留自研的 Claude Code 仿制管线。每个完整回答结束后，后台 LangMem 管理器会抽取、更新或删除 `user`、`feedback`、`project`、`reference` 四类结构化长期记忆。记忆按论文隔离，通过 LangGraph `BaseStore` 适配器持久化到现有 SQLite，并使用本地多语言 MiniLM Embedding 与余弦相似度检索；低于阈值时直接返回无匹配，因此普通论文追问在回答前不会新增模型侧记忆选择调用。用户明确要求忽略记忆时，本轮不注入任何长期记忆。SQLite 原始消息仍用于会话历史展示，但不会再被当作长期记忆召回，避免已删除信息从旧消息中重新出现。旧 SQLite 与文件记忆只进行一次兼容导入。

源码机制与运行时映射的逐项清单见 [`docs/claude-memory-port.md`](docs/claude-memory-port.md)。

每次完成正式分析后，系统都会返回一个不透明的 `analysis_id`。后端在有界的四小时内存缓存中保留该分析对应的完整 `E`/`T`/`F` 证据片段。Agent 取证、论文追问、多论文对比、早期消息召回和主题记忆召回以本机多语言 Embedding 为主排序；首次使用会下载约 240 MB 的 `paraphrase-multilingual-MiniLM-L12-v2` 到 `.paper-reader/models/`，不发送论文文本，也不需要厂商 Key。模型不可用时才回退到旧的词面检索。可通过 `PAPER_READER_MODEL_DIR` 修改缓存目录，或用 `PAPER_READER_DISABLE_EMBEDDINGS=true` 明确关闭。回答 Prompt 将论文原文证据视为最高依据，并明确区分论文事实、背景知识、长期记忆与模型推断。

当用户明确询问近期工作、相关论文或与其他论文的对比时，系统还可以使用 Semantic Scholar 提供的题录和摘要信息。该查询是可选能力，失败时会安全降级；可以设置 `SEMANTIC_SCHOLAR_API_KEY` 以使用独立 API 配额。Sample 和 Demo 结果采用确定性回复，因此无需额外调用模型也能验证完整交互流程。重新打开已保存论文时，系统会从持久化的完整证据中重新建立实时聊天证据会话。

## 论文历史

每次完成的上传默认保存在本地 `.paper-reader/` 目录。SQLite 数据库中包含论文元数据、结构化 Agent 输出、评估结果、完整证据片段、单篇与多论文对比工作区、聊天会话及不可变原始消息；原始 PDF 保存在 `.paper-reader/papers/`，分层文件记忆保存在 `.paper-reader/memory/`。再次上传同一份 PDF 时，系统会更新已有历史记录，而不是重复创建。`Recent Papers` 和顶部 History 菜单均从该数据库读取数据，因此用户刷新浏览器或重启后端后，可以直接恢复论文分析及其全部会话，无需重新上传 PDF。重新打开正式分析结果时，系统还会根据已保存证据重建追问所需的证据会话。

可以通过 `PAPER_READER_DATA_DIR` 修改全部历史数据的存储位置，或通过 `PAPER_HISTORY_DB` 指定 SQLite 文件路径。从 History 菜单删除论文时，其数据库记录、追问会话和保留的 PDF 文件都会一并删除。

原始架构说明请参阅 [CLAUDE.md](./CLAUDE.md)。

## 技术栈

- LangGraph：编排多 Agent 工作流
- OpenAI Codex Python SDK：可选地复用本机 ChatGPT 订阅执行模型推理
- PyMuPDF4LLM Layout + PyMuPDF：正文/公式/表格/图注分类，位图与矢量图区域识别及精确渲染
- FastEmbed：本地多语言语义检索；可靠度评分仍由可审计的确定性规则计算
- Pydantic v2：结构化输出 Schema 与结果校验
- 证据片段：为结论提供页码和章节依据（`E` 正文、`T` 表格、`F` 视觉图像摘要）
- FastAPI：后端 API 与静态文件托管
- React + Vite：前端论文研读工作台
