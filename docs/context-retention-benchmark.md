# 长上下文事实保留 A/B Benchmark

## 结论

本评测用于回答一个窄问题：当约 13 万 Token 的论文追问候选上下文被控制到
8K 输入预算时，当前上下文治理链路还能保留多少可精确核验的事实。

2026-07-28 使用 Qwen `qwen3.7-max` 快速模式完成三轮真实模型测试。冻结集包含
15 个独立 Gold 事实，分成 5 组问题，每组重复 3 次，因此每个基线有 15 次模型
调用和 45 个事实观测。Qwen 官方文档给出的上下文长度为 1M，能够容纳本评测的
Full-context 基线：[qwen3.7-max 模型信息](https://help.aliyun.com/zh/model-studio/qwen3-7-max)。

| 基线 | 上下文策略 | 准确率 | 数值 EM | 长期记忆 | 论文证据 | 最近状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A0 Full-context | 约 13 万 Token 全量输入 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A1 Trim-only 8K | 仅保留当前 8K 裁剪结果 | 33.3% | 25.0% | 0.0% | 0.0% | 100.0% |
| A2 Managed 8K | 8K + LangMem + 论文证据重检 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

A2 相对 A1 提高 66.7 个百分点，并在本冻结集上与 A0 的准确率持平。A2 的估算
输入从 A0 平均 130,881 Token 降至 6,982 Token，减少 94.666%。该结果证明修复后
的链路在本压力场景中同时保留了长期记忆、论文证据和最近状态；它不代表任意开放
问题都能实现无损压缩。

## 冻结协议

评测清单位于 `benchmarks/context-retention-v1.json`，SHA-256 为：

```text
b4c37f7f604e68f39d2a24b851f4bc1e0ef2b95c0db6ebba1ec10b6502d87423
```

候选上下文由以下部分确定性生成：

- 20 条历史消息，共 110,000 Token；
- 大型论文分析上下文，共 20,019 Token；
- 合计 130,019 Token，不包含系统指令和最终问题；
- 5 个 LangMem 长期事实，覆盖 `user`、`feedback`、`project`、`reference`；
- 5 个论文证据事实，覆盖数据集、准确率、Batch Size、学习率和消融结果；
- 5 个最近状态事实，覆盖 GPU、工单、Checkpoint、分支和阈值。

每组问题恰好包含一个长期记忆事实、一个论文证据事实和一个最近状态事实。模型只
返回带固定 Fact ID 的 JSON；判分采用归一化后的 Exact Match，不使用 LLM Judge。
三轮使用同一模型、快速模式和 `CHAT_TEMPERATURE=0`。

## 基线定义

### A0 Full-context

将完整 110,000-Token 历史、20,019-Token 分析上下文、当前论文检索结果和问题
一次性发给模型。它是本评测的高成本 Oracle，用于判断 Gold 问题本身是否可回答，
不是产品默认路径。

### A1 Trim-only 8K

使用当前 `build_chat_prompt` 的 8K 动态预算，但不绑定论文分析 Session 和
LangMem 会话。该基线只能依赖预算内保留下来的最近消息与压缩分析结果。

### A2 Managed 8K

使用真实产品链路：

1. 完整回合在回答后交给 LangMem manager；
2. 长期记忆写入论文级 SQLite/LangGraph Store；
3. 问题到来时执行本地语义记忆召回；复合问题拆成最多 6 个问题子句；
4. 从论文 Evidence Index 重新检索原文；
5. 按“论文证据 → 命中记忆 → 最近消息 → 分析结果”装配最终输入。

## 修复前后

LangMem manager 成功处理 20 条消息，并抽取 5/5 个长期 Gold 事实，抽取召回率为
100%，单次后台处理耗时约 14～19 秒。最初版本的 A2 准确率只有 66.7%，但失败
并不是 LangMem 没有记住。

| A2 版本 | 总准确率 | 长期记忆 | 论文证据 | 最近状态 |
| --- | ---: | ---: | ---: | ---: |
| 修复前 | 66.7% | 0.0% | 100.0% | 100.0% |
| 修复后 | 100.0% | 100.0% | 100.0% | 100.0% |

修复包含两部分：

1. **预算防饥饿**：将命中的 `<recalled_topic_memory>` 提升到论文证据之后、
   最近消息之前；其动态上限为 `min(5000, max(800, token_budget / 12))`，因此
   8K 压力配置保留最多 800 Token，默认 48K 配置保留最多 4K Token。
2. **复合问题分解召回**：完整查询继续使用 0.48 相似度阈值；当一轮包含多个
   问号子句时，对最多 6 个子句分别取 Top-1，并仅对该窄查询使用降低 0.15 的
   阈值，随后按记忆 ID 去重、分数排序并保持最终 Top-3 上限。这样没有全局降低
   单问题阈值，也不会让大量弱相关记忆进入 Prompt。

修复后 15 个独立事实在三轮中全部命中，45/45 个事实观测 Exact Match 正确，
Missing Rate 与 Unsupported Answer Rate 均为 0%。

## 延迟与 Token

| 基线 | 实际输入均值 | 冷启动 P50 | 缓存后 P50 | 总体 P95 |
| --- | ---: | ---: | ---: | ---: |
| A0 | 127,642 Token | 8.15 s | 2.17 s | 8.34 s |
| A1 | 6,889 Token | 1.30 s | 1.04 s | 1.93 s |
| A2 | 6,937 Token | 1.30 s | 1.38 s | 1.95 s |

Qwen 上下文缓存显著影响 A0 的后两轮延迟，因此简历或报告不能只比较混合后的
P50，也不能将缓存收益归因于本项目的裁剪算法。

## 复现

只检查冻结集规模，不调用远程模型：

```bash
./.venv/bin/python tools/run_context_retention_benchmark.py --dry-run
```

执行完整三轮真实模型评测：

```bash
./.venv/bin/python tools/run_context_retention_benchmark.py \
  --repeats 3 \
  --overwrite
```

运行结果写入：

```text
.paper-reader/benchmarks/context-retention-v1/results.jsonl
.paper-reader/benchmarks/context-retention-v1/report.json
```

结果目录受 `.gitignore` 保护，不包含 API Key、原始用户对话或认证信息。真实模型
调用会产生费用；发布数字前必须同时保留 manifest SHA、模型、模式、重复次数和
完整报告。

## 结论边界

- 本结果是冻结合成长上下文压力测试，不代表真实用户流量中的平均准确率。
- 15 个 Gold 事实是独立题目，三轮重复产生的 45 个观测不应写成 45 个独立样本。
- Exact Match 能稳定验证代码、数值和日期，但不能衡量开放式答案的语义质量。
- 可以表述“在该冻结压力集上准确率与全量上下文持平”，但不能扩展为真实流量平均
  准确率，也不能声称任意历史细节都绝对不丢失。
