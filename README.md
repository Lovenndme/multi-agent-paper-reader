# Multi-Agent Paper Reader

基于 LangGraph 的多 Agent 协作论文研读助手。上传学术论文 PDF，系统通过四个专职 Agent（方法解析 / 实验分析 / 批判性评审 / 总结编排）协作，输出结构化的论文研读笔记。

## Quick Start

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 运行
python main.py examples/your_paper.pdf --pretty
```

## Architecture

```
PDF → Parser → [MethodAgent | ExperimentAgent | CriticAgent] → SummaryAgent → 结构化笔记
```

详细架构说明见 [CLAUDE.md](./CLAUDE.md)。

## Tech Stack

- LangGraph (Agent orchestration)
- PyMuPDF (PDF parsing)
- Pydantic v2 (Output schema validation)
- Streamlit (Optional frontend)
