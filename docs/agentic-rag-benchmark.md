# Agentic RAG Benchmark

## 评测目标

本项目将检索控制器、证据监督和整篇论文生成分层评测，避免用一个总分混淆不同
模块的收益。评测结果必须同时报告质量、延迟、调用次数、失败和回退，不能把局部
检索命中率写成整套系统的论文解读准确率。

## 基线

### Track R：Source-anchored 检索

| 基线 | 配置 | 回答的问题 |
| --- | --- | --- |
| A0 | 词法职责静态检索 | 不使用 Embedding 时能召回多少证据 |
| A1 | FastEmbed 职责静态检索 | 本地语义检索相对词法检索的收益 |
| AQL | 改造前父块 query-aware 检索 | 分层检索相对旧实现是否有收益 |
| AQ | token 子块/父块回并 + BM25/Dense RRF | 不调用 LLM 的强 RAG 基线 |
| AQR | AQ + 本地 Cross-Encoder | 候选重排能否提高前排召回 |
| A2 | A1 seed + 结构化动作 Agentic RAG | 通用结构化规划循环是否扩大证据覆盖 |
| A3 | A1 seed + 原生工具 Agentic RAG | 原生工具调用能否降低规划开销并保持覆盖 |
| A4 | A1 seed + 自适应门控 + 原生工具 | 只在证据风险出现时规划的质量/延迟折中 |

Gold 由 PDF SHA-256、页码范围和人工核验原文引文组成。评分器按稳定的
`source_anchor` 判断命中；`E003`、`T001`、`F002` 只用于定位当前运行，不能单独
作为跨版本 Gold。

当前主指标为 Evidence Recall@5，用于约束前排证据质量；Recall@10/16 检查生产
配置最终证据预算内的覆盖。辅助指标包括 MRR、nDCG、Precision、证据字符数、
工具调用数、工具成功率、回退率、错误率及检索 P50/P95 延迟。

### Track P：完整主链路

| 基线 | 配置 |
| --- | --- |
| B0 | Hybrid 静态检索 + Method/Experiment/Critic + Evidence Supervisor/repair + Summary |
| B1 | Native Agentic RAG + 相同专业 Agent、Supervisor/repair 和 Summary |
| B2 | Adaptive Agentic RAG + 首轮静态、定向 repair、条件式 Summary 补查 |

完整主链路首先检查 Schema 成功、引用 ID 有效率、Supervisor 结果、修复次数、
公开检索 trace 和端到端延迟。只有建立人工事实 Gold 后，才计算 Grounded Fact
Precision、Recall、F1、数值准确率和 Unsupported Claim Rate。

并行调度需要另设 P0/P1：使用完全相同的三个专业 Agent、Prompt、模型和证据，
仅切换串行与并行执行。不能用单 Agent 与并行多 Agent 的耗时差宣称 DAG 加速。

## 2026-07-25 Pilot

### 环境与样本

- 代码：`774842c29cfb04f5ff2b5ee78d544a970254c9d4`
- 项目版本：`V1.6.2`
- 模型：Qwen3.7 Max，thinking
- FastEmbed：`0.8.0`
- 数据：4 篇本地公开论文，每篇 Method、Experiment、Critic 各 1 题，共 12 题
- Gold：8 个正文、2 个表格、2 个图像证据目标；所有引文均人工核对页码
- 重复：1 次；这是工程 Pilot，不是可对外泛化的正式 Benchmark
- 延迟：常驻进程热启动；FastEmbed 首次模型加载不计入逐题检索延迟

### Track R 结果

| 基线 | Recall@16 | MRR | nDCG@16 | Precision@16 | 平均延迟 | P50 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 词法静态 | 41.7% | 0.143 | 0.195 | 4.3% | 1.3 ms | 1.3 ms | 0 |
| A1 FastEmbed 静态 | 50.0% | 0.179 | 0.241 | 5.3% | 0.3 ms | 0.2 ms | 0 |
| A2 结构化 Agentic | 91.7% | 0.215 | 0.356 | 7.4% | 53.4 s | 57.8 s | 3.42 |
| A3 原生工具 Agentic | 100.0% | 0.224 | 0.381 | 8.9% | 18.2 s | 19.1 s | 1.17 |

A3 在 12/12 个 case 中命中 Gold，全部实际使用 `native` 策略，未发生传输回退、
工具错误或运行错误。相对 A1，Recall@16 提升 50.0 个百分点；在配对 case 上有
6 个改善、0 个退化，但样本规模仍然很小。相对 A2，A3 多命中 1 个 case，平均
延迟降低 65.9%，平均工具调用减少 65.9%。

A2 唯一漏检的是 Transformer 自注意力与循环层复杂度对比。A2 有 7/12 个 case
耗尽 4 步预算；A3 的所有 case 均由模型主动结束。这表明原生工具路径在当前模型
上比结构化动作回退更高效，但不能据此推断其他模型厂商也有相同结论。

### Track P 单论文结果

使用《Attention Is All You Need》对完整主链路各运行 1 次：

| 配置 | 端到端延迟 | Schema | 引用ID有效率 | 引用数 / 去重数 | 修复 | Supervisor |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B0 Hybrid | 416.4 s | 成功 | 100% | 12 / 7 | 1 | 修复后仍不足 |
| B1 Native Agentic | 820.5 s | 成功 | 100% | 39 / 16 | 1 | 修复后仍不足 |

B1 的去重引用数增加 128.6%，但端到端延迟增加 97.1%，且 Supervisor 没有从
“不足”变为“充分”。因此本次 Pilot 只能证明 Agentic 检索扩大了证据覆盖，不能
证明最终论文解读准确率提升。`coverage_score` 还出现 2/100 和 4/100；当前
Supervisor Prompt 没有明确该字段的百分制标尺，这个值不适合作为质量指标。

## 2026-07-26 Adaptive Follow-up

在同一份 4 篇论文、12 题 Gold 上增加 A4。工作树基于 `774842c29cfb`，默认
planner 仍使用 Qwen3.7 Max thinking；自适应最大 2 步。

| 配置 | Recall@16 | MRR | nDCG@16 | 平均延迟 | P50 | 触发率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 静态 | 50.0% | 0.179 | 0.241 | 0.3 ms | 0.2 ms | 0% | 0 |
| A4 Adaptive thinking | 58.3% | 0.262 | 0.324 | 0.97 s | 1.5 ms | 8.3% | 0.17 |
| A3 全量 Agentic thinking | 100.0% | 0.224 | 0.381 | 18.2 s | 19.1 s | 100% | 1.17 |

A4 只在 Transformer 的复杂度批判题触发，补回 1 个 A1 未命中的 Gold；相对 A1
Recall 提高 8.3 个百分点，相对 A3 平均检索延迟降低 94.7%，但 Recall 仍低
41.7 个百分点。这是明确的成本/召回折中，不是对 A3 的全面替代。

另做 `AGENTIC_RAG_PLANNER_MODE=fast` 消融：平均延迟为 0.30 秒，但触发题重复
调用同一工具，整体 Recall 回落到 50.0%，回退率为 8.3%。因此当前默认不启用
fast planner；该配置只保留为按模型单独验证的可选项。

完整 B2 仍使用《Attention Is All You Need》和与 B0/B1 相同的结果口径：

| 配置 | 延迟 | Schema | 有效引用率 | 引用 / 去重 | 修复 | 触发轨迹 | 规划步骤 | Supervisor |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| B2 Adaptive thinking | 659.7 s | 成功 | 100% | 15 / 9 | 1 | 2 / 5 | 3 | 67 分，修复后仍不足 |

B2 相对 B0 的延迟增加 58.4%，去重引用增加 28.6%；相对 B1 的延迟降低
19.6%，规划步骤从 20 降至 3，减少 85.0%。5 条检索轨迹中 3 条直接复用静态
证据，2 条按缺口补检索，且没有 planner fallback。Supervisor 最终仍给出 4 条
warning 并判定证据不足，因此这些数据只支持“减少无差别规划、保留部分覆盖增益”，
不支持“最终解读准确率提升”。B0/B1 使用的是旧版无明确分值标尺的 Supervisor
Prompt，其历史 `coverage_score` 不与 B2 的 67 分直接比较。

## 2026-07-26 Expanded Retrieval Follow-up

### 扩容与口径

为检查 12 题 Pilot 的小样本偏差，在原 4 篇论文基础上新增 8 篇公开论文：
[BERT](https://arxiv.org/abs/1810.04805)、
[ResNet](https://arxiv.org/abs/1512.03385)、
[Adam](https://arxiv.org/abs/1412.6980)、
[LoRA](https://arxiv.org/abs/2106.09685)、
[RAG](https://arxiv.org/abs/2005.11401)、
[DDPM](https://arxiv.org/abs/2006.11239)、
[U-Net](https://arxiv.org/abs/1505.04597) 和
[CLIP](https://arxiv.org/abs/2103.00020)。扩展后共 12 篇、36 题，Method、
Experiment、Critic 各 12 题；每题绑定当前解析索引中的原文引文、页码和 PDF
SHA-256。

本轮仍使用 Qwen3.7 Max thinking、FastEmbed `0.8.0`、常驻进程热启动和 1 次
重复。除与生产证据上限一致的 Recall@16 外，新增更严格的 Recall@5，检查 Gold
是否真正进入前排。本轮仍是检索层工程试验，不代表完整回答准确率，也不能替代
至少 3 次重复的正式 Benchmark。

原 A1 只使用 Agent 职责描述作为向量 query，未直接使用当前问题。为避免给
Agentic RAG 设置过弱对照，本轮补充 AQ：使用完全相同的论文索引和 FastEmbed，
直接以用户问题执行一次语义+词法混合检索，不调用 LLM。

### 结果

| 基线 | Recall@5 | Recall@16 | MRR@5 | nDCG@5 | 平均延迟 | P50 | 触发/回退 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 词法职责静态 | 25.0% | 41.7% | 0.165 | 0.186 | 1.1 ms | 0.7 ms | 0% |
| A1 向量职责静态 | 16.7% | 30.6% | 0.094 | 0.112 | 0.2 ms | 0.1 ms | 0% |
| AQ 单次 query-aware 混合检索 | **77.8%** | **86.1%** | **0.528** | **0.579** | **4.4 ms** | 3.8 ms | 0% |
| A3 全量原生工具 Agentic | 58.3% | 77.8% | 0.421 | 0.462 | 15.61 s | 13.63 s | 5.6% 回退 |
| A4 Adaptive Agentic | 27.8% | 41.7% | 0.165 | 0.193 | 1.51 s | 2.3 ms | 11.1% 触发 |

AQ 在 36 题中的 Recall@5 为 28/36，A3 为 21/36。配对比较中，A3 相对 AQ
改善 1 题、持平 27 题、退化 8 题；双侧 exact McNemar 检验 `p=0.039`。在
Recall@16 上，AQ 为 31/36，A3 为 28/36，A3 改善 0 题、持平 33 题、退化
3 题；当前样本下差异未达到显著水平（`p=0.25`）。

AQ 的 Recall@5 在 Method、Experiment、Critic 上分别为 75.0%、75.0% 和
83.3%；A3 分别为 66.7%、41.7% 和 66.7%。因此 AQ 的领先不是由单一 Agent
类别贡献。A3 相对弱 A1 的 Recall@16 确实提高 47.2 个百分点，但 AQ 证明这个
对比不足以支撑“Agentic 优于传统 RAG”的结论。

本轮问题大多只有一个 Gold 证据锚点，且只运行 1 次；它更适合回答“单证据检索
是否需要规划”，尚不能回答跨章节、多证据组合、图表联动和多跳任务是否受益于
Agentic RAG。

## 2026-07-27 Independent Frozen Test

### 数据冻结与防泄漏

前述 12 篇、36 题扩展集已参与方案分析，因此只能作为开发集。本轮另外选取 8 篇
此前未进入调参的公开论文：
[GPT-3](https://arxiv.org/abs/2005.14165)、
[ViT](https://arxiv.org/abs/2010.11929)、
[Batch Normalization](https://arxiv.org/abs/1502.03167)、
[Dropout](https://arxiv.org/abs/1207.0580)、
[VAE](https://arxiv.org/abs/1312.6114)、
[DQN](https://arxiv.org/abs/1312.5602)、
[Word2Vec](https://arxiv.org/abs/1301.3781) 和
[GAN](https://arxiv.org/abs/1406.2661)。

每篇分别设置 Method、Experiment、Critic 各 1 题，共 24 题。每题的 PDF
SHA-256、页码、原文引文和逻辑 `fact_id` 均在执行检索前冻结；manifest SHA-256
为 `5de20bad11b4dde5b34c4c962fa3fb8dcd8799b8e98d07e2bbef70024a2f16c7`。
同一事实允许多个经人工核验的等价原文锚点，命中任一锚点即算召回一个事实，避免
把同义证据错误计为多个必选 Gold。全部配置运行 3 次，共 72 个运行记录；重复只用于
检查确定性与延迟，不把独立样本量从 24 夸大成 72。

### 检索链路消融

默认 Embedding 为
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，AQR 使用
`Xenova/ms-marco-MiniLM-L-6-v2` 对 Top-24 候选重排。延迟为同一常驻进程预热后的
单题检索耗时，不包含首次模型下载；主指标采用更严格的 Recall@5。

| 基线 | 关键差异 | Recall@5 | Recall@10 | MRR | nDCG@5 | P50 | P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AQL | 改造前父块 query-aware 检索 | 62.5% | 83.3% | 0.439 | 0.486 | 0.8 ms | 7.0 ms |
| AQ | PDF 归一化 + token 子块/父块回并 + BM25/Dense RRF | 79.2% | 100.0% | 0.625 | 0.668 | 14.7 ms | 79.0 ms |
| AQR | AQ + 本地 Cross-Encoder 重排 | **91.7%** | **100.0%** | **0.705** | **0.759** | **401.9 ms** | **726.3 ms** |

AQR 命中 22/24 题；相对 AQL 的 Recall@5 提升 29.2 个百分点，相对不重排的 AQ
提升 12.5 个百分点。代价是相对 AQ 增加约 387 ms P50 延迟。Cross-Encoder
不可用时会保留 AQ 排序，因此模型下载、加载或推理失败不会阻断分析。

### Embedding 对照

为避免把 Cross-Encoder 的收益误归因于 Embedding，更换模型时只运行 AQ：

| Embedding | 语言/体积 | Recall@5 | Recall@10 | MRR | P50 | P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| multilingual MiniLM-L12-v2 | 多语言 / 0.22 GB | 79.2% | 100.0% | 0.625 | 14.7 ms | 79.0 ms |
| BGE-base-en-v1.5 | 英文 / 0.21 GB | 87.5% | 95.8% | 0.683 | 15.7 ms | 79.9 ms |
| multilingual-e5-large | 多语言 / 2.24 GB | **91.7%** | **100.0%** | **0.771** | 14.9 ms | 91.5 ms |

BGE 在这组全英文问题上提高前排召回，但牺牲了项目需要的跨语言检索能力，且
Recall@10 反而少 1 题，因此不能仅凭这组英文测试替换多语言默认模型。E5-large
在不重排时已达到轻量 MiniLM + Cross-Encoder 的 Recall@5，并取得本轮最高 MRR；
但模型体积约为 MiniLM 的 10.2 倍，默认 256 batch 在测试机触发明显内存换页。
将编码批量降至 16 后才稳定完成全量测试，因此它保留为高质量可选配置，不作为
跨平台默认下载。E5-large 再叠加 Cross-Encoder 后 Recall@5 仍为 91.7%，P50
增至 498.6 ms，没有召回收益，所以高质量配置只使用 AQ，不重复重排。

## 当前生产决策

默认轻量主链路采用 AQR：先执行确定性的分层混合检索和本地重排；内存充足时可将
Embedding 切换为 E5-large，并按机器资源下调编码 batch。Adaptive Agentic
只在低检索置信度、跨章节、多证据、精确数值、公式、图表联动或 Supervisor 定向
修复时触发；全量 Agentic 继续作为 A/B 配置，不作为默认路径。这样保留工具规划
处理复杂证据缺口的能力，同时不让普通单证据问题支付十几秒规划开销。

本轮结果只证明 source-anchored 检索召回提升，不代表最终论文解读准确率提高。
在人工标注整篇输出的原子事实与支持证据前，不发布 Grounded Fact F1，也不把
Recall@5 改写成“回答准确率”。

## 复现

实际论文、Gold 引文和运行结果保存在本机 `.paper-reader/benchmarks/`，不进入
Git。示例命令：

```bash
./.venv/bin/python tools/run_agentic_rag_benchmark.py \
  .paper-reader/benchmarks/agentic-rag-frozen/manifest.jsonl \
  --output .paper-reader/benchmarks/agentic-rag-frozen/results.jsonl \
  --report .paper-reader/benchmarks/agentic-rag-frozen/report-k5.json \
  --baselines A0,A1,AQL,AQ,AQR --repeat 3 --k 5

./.venv/bin/python tools/benchmark_agentic_rag.py \
  .paper-reader/benchmarks/agentic-rag-frozen/results.jsonl \
  --k 10 \
  --output .paper-reader/benchmarks/agentic-rag-frozen/report-k10.json

PAPER_READER_EMBEDDING_BATCH_SIZE=16 \
EMBEDDING_MODEL=intfloat/multilingual-e5-large \
./.venv/bin/python tools/run_agentic_rag_benchmark.py \
  .paper-reader/benchmarks/agentic-rag-frozen/manifest.jsonl \
  --output .paper-reader/benchmarks/agentic-rag-frozen/results-e5.jsonl \
  --baselines AQ --repeat 3 --k 5

./.venv/bin/python tools/run_pipeline_benchmark.py <paper_history_id> \
  --output .paper-reader/benchmarks/agentic-rag-pilot/pipeline-b2.json \
  --label B2 --rag-mode adaptive --tool-strategy native
```
