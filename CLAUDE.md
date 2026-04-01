# Multi-Agent Paper Reader

## 项目概述

多Agent论文研读助手：用户上传学术论文PDF，系统通过4个专职Agent协作，输出结构化的论文研读笔记。

## 架构设计

### Agent 编排流程

```
PDF Upload → PDF Parser (章节分割)
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
 Method      Experiment   Critical
 Agent        Agent       Review Agent
    ↓           ↓           ↓
    └───────────┼───────────┘
                ↓
         Summary Agent
                ↓
        结构化论文笔记输出
```

前三个Agent并行执行，结果汇总给Summary Agent。使用LangGraph编排工作流。

### 四个Agent职责

1. **MethodAgent** - 提取研究方法、模型架构、技术创新点，与已有方法的区别
2. **ExperimentAgent** - 提取数据集、评估指标、实验结果、性能对比数据
3. **CriticAgent** - 评估创新性、指出局限性、提出改进方向
4. **SummaryAgent** - 整合前三者输出，生成结构化笔记（一句话概括 / 核心贡献 / 方法细节 / 实验亮点 / 局限与展望）

### 输出格式

每个Agent的输出用Pydantic Model约束为JSON Schema，确保下游整合稳定。

## 目录结构

```
multi-agent-paper-reader/
├── CLAUDE.md              # 本文件 - 项目规范
├── requirements.txt       # Python依赖
├── .env.example           # 环境变量模板
├── main.py                # CLI入口
├── app.py                 # Streamlit前端（可选）
├── core/
│   ├── __init__.py
│   ├── pdf_parser.py      # PDF解析与章节分割
│   ├── graph.py           # LangGraph工作流编排
│   └── schemas.py         # Pydantic输出模型定义
├── agents/
│   ├── __init__.py
│   ├── method_agent.py    # 方法解析Agent
│   ├── experiment_agent.py # 实验分析Agent
│   ├── critic_agent.py    # 批判性评审Agent
│   └── summary_agent.py   # 总结编排Agent
├── prompts/
│   ├── method.txt         # MethodAgent提示词
│   ├── experiment.txt     # ExperimentAgent提示词
│   ├── critic.txt         # CriticAgent提示词
│   └── summary.txt        # SummaryAgent提示词
├── utils/
│   ├── __init__.py
│   └── llm.py             # LLM调用封装
├── tests/                 # 测试用例
├── examples/              # 示例论文PDF
└── docs/                  # 项目文档
```

## 技术栈

- **Agent编排**: LangGraph
- **LLM调用**: langchain-openai（兼容OpenAI和Qwen API）
- **PDF解析**: PyMuPDF (fitz)
- **输出约束**: Pydantic v2
- **前端(可选)**: Streamlit

## 开发顺序

1. `core/schemas.py` - 定义四个Agent的输出数据模型
2. `core/pdf_parser.py` - PDF文本提取与章节分割
3. `utils/llm.py` - LLM调用封装
4. `prompts/*.txt` - 四个Agent的提示词
5. `agents/*.py` - 四个Agent实现
6. `core/graph.py` - LangGraph工作流编排
7. `main.py` - CLI入口
8. 测试与调优
9. `app.py` - Streamlit前端（如果有时间）

## 关键设计决策

- 前三个Agent并行执行以提高速度，用LangGraph的fan-out/fan-in模式
- 每个Agent的提示词单独存放在prompts/目录，方便迭代调优
- LLM调用统一封装，切换模型只需改.env配置
- PDF按章节分割后，各Agent只接收与自己职责相关的章节，减少token消耗
