# Multi-Agent Paper Reader

[English](./README.md) | **简体中文**

一个在本机运行的 AI 论文研读工作台。上传 PDF 后，系统会结合正文、表格和图像证据，由多个专职 Agent 分析论文的方法、实验与局限，并生成可追问、可对比、可追溯的结构化笔记。

![Multi-Agent Paper Reader 工作台](./docs/assets/paper-reader-workspace.png)

## 能做什么

- **多 Agent 论文分析**：Method、Experiment、Critic 和 Summary Agent 协同生成研读笔记。
- **图表与公式理解**：识别正文、表格、公式、图注、位图和矢量图，并保留高分辨率导出能力。
- **基于原文追问**：围绕当前论文连续提问，回答由章节、页码和证据片段约束。
- **多论文对比**：选择 2～4 篇历史论文，对比方法、实验、结论和研究空白。
- **可解释评估**：分别展示论文创新性评分与分析可靠度，保留评分依据和警告。
- **本地历史与记忆**：论文、分析、会话和长期记忆保存在本机 `.paper-reader/`。
- **灵活模型路由**：支持 GLM、DeepSeek、OpenAI、Qwen、Doubao、Anthropic、Kimi、自定义中转站以及本机 Codex 订阅。

## 界面预览

模型路由与 Codex 订阅：

![模型路由与 Codex 订阅](./docs/assets/codex-model-routing.png)

多论文对比工作区：

![多论文对比工作区](./docs/assets/comparison-workspace.png)

## 快速开始

要求 Python 3.10+。从源码安装时建议使用 Node.js 22；正式 Release ZIP 已包含构建完成的前端。

### Windows

```powershell
git clone https://github.com/Lovenndme/multi-agent-paper-reader.git
cd multi-agent-paper-reader
powershell -ExecutionPolicy Bypass -File .\scripts\update.ps1
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### macOS / Linux

```bash
git clone https://github.com/Lovenndme/multi-agent-paper-reader.git
cd multi-agent-paper-reader
bash ./scripts/update.sh
./.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)，进入 **Settings** 配置模型，然后上传 PDF。

不使用 Git 的用户也可以从 [GitHub Releases](https://github.com/Lovenndme/multi-agent-paper-reader/releases/latest) 下载正式发行包。

## 使用 Codex 订阅

项目可以在本机复用用户自己的 ChatGPT/Codex 订阅，无需 OpenAI API Key：

1. 打开 **Settings → Codex 订阅**。
2. 复用已有本机登录，或使用浏览器/设备码完成登录。
3. 在模型路由中选择 Codex 模型和推理强度。

Codex 登录由官方 runtime 处理，网页不会读取、保存或返回 token。该功能只适用于通过 `localhost` 使用的本机单用户部署，不适用于公网多人服务。

详细的模型目录、工具权限和安全边界见 [Codex 订阅接入说明](./docs/codex-subscription.md)。

## 数据与隐私

- 原始 PDF、分析结果、对话和记忆默认保存在本机 `.paper-reader/`。
- API Key 仅写入本机 `.env`，保存后不会回显到网页。
- 使用第三方模型厂商时，相关论文内容会发送给用户选择的服务商。
- Codex 订阅、API 厂商和自定义中转站之间不会自动改道，以用户选择的模型路由为准。

## 更新

先停止旧服务，再运行：

```bash
# macOS / Linux
bash ./scripts/update.sh
```

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\update.ps1
```

更新脚本会保留 `.env` 和 `.paper-reader/`，重新安装依赖、构建前端，并强制检查前后端版本一致性。更新完成后需要重新启动服务。

## CLI

```bash
./.venv/bin/python main.py path/to/paper.pdf --pretty
```

Windows 使用：

```powershell
.\.venv\Scripts\python.exe main.py path\to\paper.pdf --pretty
```

## 技术栈

FastAPI · React · Vite · LangGraph · Pydantic · PyMuPDF4LLM · FastEmbed · OpenAI Codex Python SDK

开发、测试和发布规范见 [AGENTS.md](./AGENTS.md)。版本更新记录见 [Releases](https://github.com/Lovenndme/multi-agent-paper-reader/releases)。
