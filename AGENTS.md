# Multi-Agent Paper Reader 开发规范

本文件适用于仓库根目录及其所有子目录。`frontend-prototype/AGENTS.md`
包含前端原型的补充设计约束；修改该目录时，应同时遵守两份文件。

## 项目现状

- 后端：Python、FastAPI、LangGraph、Pydantic、PyMuPDF/PyMuPDF4LLM。
- 前端：`frontend-prototype/` 下的 React 19 + Vite 6 应用。
- Web 入口：`app.py`，同时提供 API 并托管生产前端。
- CLI 入口：`main.py`。
- 当前版本的唯一代码来源：`core/settings.py` 中的 `PROJECT_VERSION`。
- 本地持久化数据默认位于 `.paper-reader/`，任何更新都不得破坏该目录。
- 模型路由包括多个 API 厂商和仅限本机单用户使用的 Codex 订阅。

主要处理链路：

```text
PDF / HTTP -> PaperAnalysisOrchestrator
    -> 解析与版面/视觉提取 -> 证据索引
    -> LangGraph（专业 Agent 并行与 Summary 汇合）
    -> Agent Harness（检索、生命周期、进度、校验）
    -> Agent Runtime（模型调用、流式适配、重试）
    -> Method / Experiment / Critic Agent
    -> Summary Agent -> 结构化分析与论文追问
```

`core/analysis_orchestrator.py` 是整篇论文任务的唯一应用编排入口，负责解析、
视觉降级、证据、LangGraph 工作流、评估、持久化和最终任务状态。
`app.py` 的分析接口只负责 HTTP 校验、调用 Orchestrator 和 NDJSON 序列化；
不得在接口层重新实现 Agent 并行、汇总或持久化流程。

Agent 的声明式契约位于各自 `agents/*_agent.py` 的 `AgentSpec` 中。Web、
LangGraph 和兼容入口必须通过 `core/agent_harness.py` 调用，
不得绕过 Harness 直接调用 `utils/llm.py`。Runtime 与厂商适配边界详见
`docs/agent-harness.md`，完整任务边界详见 `docs/analysis-orchestrator.md`。

修改结构化输出时，必须同步检查 `core/schemas.py`、相关 Agent、提示词、
汇总逻辑、API 响应、前端渲染和测试，不能只修改其中一层。

## 安装与运行

要求 Python 3.10+。源码工作区构建前端还需要 Node.js 22 和 npm。

macOS / Linux：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
npm --prefix frontend-prototype ci
npm --prefix frontend-prototype run build
./.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix frontend-prototype ci
npm --prefix frontend-prototype run build
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

本地配置可由 `.env.example` 复制为 `.env`，也可以在 Settings 中完成。
不得提交真实 API Key、Codex token、认证缓存或包含用户隐私的数据。

## 修改原则

- 先阅读相关实现和测试，再修改代码；保留工作区内与当前任务无关的用户改动。
- 使用 `rg` / `rg --files` 搜索；避免无关的全仓库机械重写。
- 不要将某个 API 厂商写死到通用分析流程中；模型能力和路由以
  `core/model_providers.py`、`core/settings.py` 与实际运行时目录为准。
- Codex 订阅必须保持本机认证边界：网页不得读取、保存或返回 token。
- PDF 视觉模型输入和面向用户导出的高分辨率图像是两条不同路径。不要为了模型
  预览而覆盖或降低原始/导出图像分辨率，也不要在超过安全上限时静默降采样。
- `.paper-reader/`、`.env`、上传论文、生成图片、缓存和构建产物不得提交到 Git。
- `frontend-prototype/dist/` 不跟随普通源码提交；正式发行包由构建和打包流程加入
  已验证的生产前端。
- 如果行为或安装方式发生变化，同步更新 `README.md` 与 `README.zh-CN.md`。

## 验证要求

后端测试：

```bash
./.venv/bin/python -m pytest -q
```

Windows 使用 `.\.venv\Scripts\python.exe -m pytest -q`。

前端生产构建：

```bash
npm --prefix frontend-prototype ci
npm --prefix frontend-prototype run build
```

影响 Web、设置页、模型路由或发行内容的修改，至少还要验证：

```bash
curl -fsS http://127.0.0.1:8000/api/health
```

健康检查必须同时满足后端正常、`frontend_dist=true` 且
`frontend_version_match=true`。不能仅凭页面能打开就判定更新成功。

每次推送 `main` 后，`.github/workflows/ci.yml` 会在 Ubuntu、Windows 和 macOS
上安装依赖、构建前端、核对前后端版本并运行完整测试。涉及跨平台脚本时，不得只在
当前系统验证。

## 前后端一致性门禁

- 构建前端时，Vite 会生成 `frontend-prototype/dist/build-meta.json`。
- 其中 `project_version` 必须等于 `core/settings.py` 的 `PROJECT_VERSION`。
- 元数据缺失、损坏或版本不一致都属于发布阻断问题。
- 修改前端后必须重新执行生产构建；不得让后端继续托管旧的 `dist`。
- 更新流程应使用 `scripts/update.sh` 或 `scripts/update.ps1`，并保留用户的
  `.env` 和 `.paper-reader/` 数据。

## 正式发布

普通代码提交不自动构成正式发行。正式版本必须完成以下步骤：

1. 更新 `PROJECT_VERSION`，并为相同版本添加中文发行说明
   `docs/releases/<VERSION>.md`。
2. 构建生产前端，执行完整测试，并确认前后端版本一致。
3. 使用 `tools/build_release_package.py` 生成正式 ZIP 和 SHA-256 校验文件。
4. 创建并推送带说明的 annotated tag；tag、`PROJECT_VERSION`、构建元数据和
   发行说明文件名必须完全一致。
5. 通过 `.github/workflows/release.yml` 发布 GitHub Release，并确认附件可用。
6. GitHub Release 的标题和正文默认使用中文；发布后等待跨平台 CI 全部通过。

打包逻辑只能收录受 Git 跟踪的源码和本次验证过的生产前端，不得包含 `.env`、
`.paper-reader/`、用户论文、认证信息、测试缓存或本地输出。

## Git 协作

- 提交前检查 `git status` 和实际 diff，只暂存本次任务相关文件。
- 未经明确授权，不覆盖、回滚或删除其他人的本地修改。
- 提交信息应简洁描述实际变化。
- 是否创建分支、PR、tag 或 Release，以当前用户要求和仓库维护者的发布规则为准。

## 文档维护

这份 `AGENTS.md` 是仓库级 AI 开发规范的唯一来源。不要再创建内容重复的
`AGENT.md` 或 `CLAUDE.md`。当架构、命令、测试或发布流程发生变化时，在同一次
提交中更新本文件，避免自动化工具继续依据过时说明工作。
