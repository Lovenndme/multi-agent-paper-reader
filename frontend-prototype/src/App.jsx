import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  IconAlertCircle,
  IconArrowDown,
  IconBook2,
  IconBrain,
  IconChartBar,
  IconCheck,
  IconChevronDown,
  IconChevronRight,
  IconClock,
  IconCloudUpload,
  IconCopy,
  IconCpu,
  IconDownload,
  IconExternalLink,
  IconEye,
  IconEyeOff,
  IconFileAnalytics,
  IconFileDescription,
  IconFileTypePdf,
  IconGripVertical,
  IconHistory,
  IconKey,
  IconListDetails,
  IconLoader2,
  IconMarkdown,
  IconMessageCircle,
  IconPencil,
  IconPlus,
  IconPhoto,
  IconQuote,
  IconRefresh,
  IconSearch,
  IconSend,
  IconSettings,
  IconShare3,
  IconShieldCheck,
  IconSparkles,
  IconTrash,
  IconX,
} from "@tabler/icons-react";
import avatarUrl from "./assets/avatar.png";
import { ComparisonWorkspace, comparisonMarkdownFromData } from "./ComparisonWorkspace.jsx";
import { ModelCallTrace } from "./ModelCallTrace.jsx";
import { useChatAutoScroll } from "./useChatAutoScroll.js";
import { useResizableChatDrawer } from "./useResizableChatDrawer.js";

const ChatMarkdown = lazy(() => import("./ChatMarkdown.jsx").then((module) => ({ default: module.ChatMarkdown })));
const conversationTitleRefreshDelays = [1200, 4000, 9000, 18000];

const tabs = ["概览", "方法", "实验", "批判性评审", "最终笔记"];
const emptyAgentStates = {
  method: "waiting",
  experiment: "waiting",
  critic: "waiting",
  summary: "waiting",
};

const emptyAgentStreams = {
  method: "",
  experiment: "",
  critic: "",
  summary: "",
};

const completeAgentStates = {
  method: "complete",
  experiment: "complete",
  critic: "complete",
  summary: "complete",
};

const defaultSettingsRouting = {
  text_provider: "zhipu",
  text_model: "glm-5.2",
  vision_enabled: true,
  vision_provider: "zhipu",
  vision_model: "glm-5v-turbo",
};

const agentStepLabels = ["阅读章节", "提取洞察", "完成输出"];

const sampleAnalysis = {
  mode: "sample",
  paper: {
    title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    filename: "rag-paper.pdf",
    pages: 21,
    sections_count: 7,
    size_bytes: 1887436,
    sections: [
      { title: "Abstract" },
      { title: "Introduction" },
      { title: "Retrieval Model" },
      { title: "Generator" },
      { title: "Experiments" },
      { title: "Ablations" },
      { title: "Conclusion" },
    ],
  },
  method_output: {
    research_problem:
      "参数化语言模型将知识存储在模型权重中，导致事实更新困难，并可能降低其在知识密集型任务上的可靠性。",
    proposed_method:
      "RAG 将神经检索器与生成器结合。检索器从外部语料库中获取 top-k 段落，生成器则同时基于输入提示和检索证据进行生成。",
    key_components: [
      "稠密段落检索器",
      "Seq2seq 生成器",
      "Top-k 边缘化",
      "外部 Wikipedia 知识库",
    ],
    innovations: [
      "参数化与非参数化记忆的混合架构",
      "以检索结果为条件的生成机制",
      "端到端检索与生成联合训练",
    ],
    differences_from_prior:
      "与闭卷生成不同，RAG 将事实证据保存在模型外部，并在推理时动态检索。",
    implementation_details:
      "采用稠密检索器和生成器，并对多个检索段落进行边缘化计算。",
  },
  experiment_output: {
    datasets: ["Natural Questions", "TriviaQA", "WebQuestions", "FEVER"],
    metrics: ["Exact Match", "Accuracy"],
    main_results:
      "该方法在多个基准上提升了事实问答和事实验证性能，其中需要显式外部证据的任务提升最明显。",
    comparison_with_baselines:
      "RAG 相较 BART、T5 及检索基线取得更好结果，说明检索与生成具有互补作用。",
    ablation_study:
      "增加检索段落数量和提升检索器质量都能改善下游任务准确率。",
    notable_findings: [
      "对多个段落进行边缘化能够提升模型鲁棒性。",
      "检索器质量与下游性能高度相关。",
    ],
  },
  critic_output: {
    novelty_score: 4,
    novelty_justification:
      "该工作提出了适用于事实生成的实用混合记忆架构，具有较强影响力和创新性。",
    novelty_dimensions: [
      {
        dimension: "problem_originality",
        score: 4,
        reason: "将知识更新与事实可靠性明确为参数化语言模型的核心限制。",
        evidence_ids: [],
      },
      {
        dimension: "method_originality",
        score: 4,
        reason: "将稠密检索、生成器与 top-k 边缘化整合为可联合训练的框架。",
        evidence_ids: [],
      },
      {
        dimension: "prior_work_difference",
        score: 4,
        reason: "相较闭卷生成，推理时能够动态使用外部、可更新的证据。",
        evidence_ids: [],
      },
      {
        dimension: "generality",
        score: 3,
        reason: "框架适用于多种知识密集型任务，但依赖外部语料与检索质量。",
        evidence_ids: [],
      },
    ],
    strengths: [
      "清晰界定参数化知识与非参数化知识。",
      "在问答和事实验证任务上进行了充分实验。",
      "通过外部索引保持知识可编辑，架构具有实用性。",
    ],
    limitations: [
      "性能高度依赖检索器质量和语料覆盖范围。",
      "与闭卷生成相比，延迟和内存占用可能增加。",
      "证据归因质量有所改善，但问题尚未完全解决。",
    ],
    potential_improvements: [
      "加入更强的引用监督。",
      "采用动态检索深度。",
      "为外部语料库增加时效性检查。",
    ],
    broader_impact:
      "检索增强系统能够改善事实性，但也可能继承检索语料中的偏见与覆盖缺口。",
  },
  summary_output: {
    one_sentence_summary:
      "RAG 提出了一种检索增强的序列到序列框架，将参数化知识与非参数化记忆结合，以提升知识密集型 NLP 任务的事实准确性。",
    core_contributions: [
      "提出通用框架，使用面向 Wikipedia 的稠密检索器增强 seq2seq 模型。",
      "在多个数据集的开放域问答与事实验证任务上取得稳定提升。",
      "提供端到端可微模型，实现高效检索与 top-k 段落边缘化。",
      "通过消融实验验证检索器质量、段落数量和边缘化策略的影响。",
    ],
    method_highlights:
      "该方法将稠密检索与生成结合，使模型能够依据检索证据作答，而不是仅依赖模型权重。",
    experiment_highlights:
      "实验结果表明，该方法在开放域问答和事实验证基准上取得了广泛提升。",
    limitations_and_future_work:
      "该方法依赖检索器质量、语料时效性以及推理阶段的检索成本。",
    reading_notes:
      "RAG 可以理解为检索系统与生成式语言模型之间的桥梁：先检索，再基于证据生成，并尽可能将知识保存在模型外部。",
  },
  assessment: {
    novelty: {
      score: 3.9,
      label: "创新性较高",
      dimensions: [
        { dimension: "problem_originality", score: 4, reason: "问题定义具有明确的新视角。", evidence_ids: [] },
        { dimension: "method_originality", score: 4, reason: "方法组合与训练机制具有原创性。", evidence_ids: [] },
        { dimension: "prior_work_difference", score: 4, reason: "与闭卷生成存在实质差异。", evidence_ids: [] },
        { dimension: "generality", score: 3, reason: "具备跨任务潜力，但仍受检索系统约束。", evidence_ids: [] },
      ],
      warnings: [],
    },
    reliability: {
      score: 39,
      raw_score: 46,
      score_cap: 39,
      level: "low",
      label: "低",
      breakdown: { parsing: 12, coverage: 12, citations: 12, output_integrity: 10 },
      warnings: ["当前为示例数据，未运行真实论文解析与证据核验。"],
    },
  },
};

function formatHistoryAge(value) {
  const timestamp = Date.parse(value || "");
  if (!Number.isFinite(timestamp)) return "已保存";
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 60) return "刚刚";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)} 天前`;
  return new Date(timestamp).toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
}

function historyPaperMeta(item) {
  const details = [item.filename];
  if (item.pages) details.push(`${item.pages} 页`);
  return details.filter(Boolean).join(" · ");
}

function HistoryPaperButton({ item, active, disabled, onOpen }) {
  return (
    <button
      className={`history-paper-button ${active ? "active" : ""}`}
      type="button"
      disabled={disabled}
      onClick={() => onOpen(item)}
    >
      <IconFileTypePdf className="pdf-icon" size={28} stroke={1.5} />
      <span>
        <strong title={item.title}>{item.title}</strong>
        <small>{historyPaperMeta(item)}</small>
      </span>
      <em>{formatHistoryAge(item.updated_at)}</em>
    </button>
  );
}

const agentBase = [
  {
    id: "method",
    name: "Method Agent",
    icon: IconBrain,
    accent: "blue",
    summary: "Maps architecture and methods.",
    x: 50,
    y: 48,
  },
  {
    id: "experiment",
    name: "Experiment Agent",
    icon: IconChartBar,
    accent: "sky",
    summary: "Extracts benchmarks and results.",
    x: 23,
    y: 218,
  },
  {
    id: "critic",
    name: "Critic Agent",
    icon: IconSearch,
    accent: "indigo",
    summary: "Reviews assumptions and limitations.",
    x: 77,
    y: 218,
  },
  {
    id: "summary",
    name: "Summary Agent",
    icon: IconSparkles,
    accent: "violet",
    summary: "Synthesizes notes and key takeaways.",
    x: 50,
    y: 468,
  },
];

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "Pending";
  if (bytes < 1024 * 1024) return `${Math.max(bytes / 1024, 1).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const chapterTitleMap = {
  abstract: "摘要",
  introduction: "引言",
  "related work": "相关工作",
  background: "研究背景",
  "research background": "研究背景",
  motivation: "研究动机",
  preliminaries: "预备知识",
  "problem formulation": "问题定义",
  method: "方法",
  methodology: "方法",
  model: "模型",
  "retrieval model": "检索模型",
  approach: "方法",
  framework: "框架",
  architecture: "模型架构",
  "model architecture": "模型架构",
  generator: "生成器",
  "encoder and decoder stacks": "编码器与解码器堆栈",
  attention: "注意力机制",
  "scaled dot-product attention": "缩放点积注意力",
  "multi-head attention": "多头注意力",
  "applications of attention in our model": "注意力在模型中的应用",
  "position-wise feed-forward networks": "逐位置前馈网络",
  "embeddings and softmax": "词嵌入与 Softmax",
  "positional encoding": "位置编码",
  "why self-attention": "为什么使用自注意力",
  experiments: "实验",
  experiment: "实验",
  "experimental setup": "实验设置",
  "experimental results": "实验结果",
  "implementation details": "实现细节",
  "hyperparameter settings": "超参数设置",
  "comparison with state-of-the-art": "与先进方法对比",
  ablations: "消融实验",
  ablation: "消融实验",
  evaluation: "评估",
  results: "实验结果",
  training: "训练",
  "training data and batching": "训练数据与批处理",
  "hardware and schedule": "硬件与训练计划",
  optimizer: "优化器",
  regularization: "正则化",
  "label smoothing": "标签平滑",
  "machine translation": "机器翻译",
  "model variations": "模型变体",
  "english constituency parsing": "英语成分句法分析",
  discussion: "讨论",
  analysis: "分析",
  limitations: "局限性",
  "future work": "未来工作",
  conclusion: "结论",
  conclusions: "结论",
  acknowledgments: "致谢",
  acknowledgements: "致谢",
  references: "参考文献",
  "full paper": "全文",
};

function cleanChapterTitle(chapter, index) {
  const candidate = chapter?.display_title || chapter?.title || "";
  const compact = String(candidate).replace(/\s+/g, " ").trim();
  const stripped = compact.replace(/^[\d一二三四五六七八九十]+(?:[\.\d]*)[\.\、\s]+/, "");
  const normalized = stripped.toLowerCase().replace(/[.:：-]+$/g, "").trim();

  if (chapterTitleMap[normalized]) return chapterTitleMap[normalized];
  if (normalized.startsWith("appendix")) return "附录";
  if (!stripped || stripped.includes("�")) return `章节 ${index + 1}`;

  const letters = stripped.match(/[A-Za-z\u4e00-\u9fff]/g) || [];
  const symbols = stripped.match(/[^A-Za-z0-9\u4e00-\u9fff\s.\-:/&]/g) || [];
  if (letters.length < 2 || symbols.length / Math.max(stripped.length, 1) > 0.16) {
    return `章节 ${index + 1}`;
  }

  return stripped.length > 48 ? `${stripped.slice(0, 45).trim()}...` : stripped;
}

function chapterMeta(chapter) {
  const start = Number.isFinite(chapter?.page_start) ? chapter.page_start + 1 : null;
  const end = Number.isFinite(chapter?.page_end) ? chapter.page_end + 1 : start;
  const pageText = start ? (end && end !== start ? `第 ${start}-${end} 页` : `第 ${start} 页`) : "页码待确认";
  const chars = Number.isFinite(chapter?.chars) ? `约 ${Math.max(Math.round(chapter.chars / 100) * 100, 100)} 字` : "内容已识别";
  return `${pageText} · ${chars}`;
}

const chapterAgentKeywords = {
  method: ["abstract", "introduction", "method", "model", "approach", "framework", "architecture", "algorithm", "摘要", "引言", "方法", "模型", "框架", "架构"],
  experiment: ["experiment", "evaluation", "result", "analysis", "ablation", "dataset", "benchmark", "实验", "评估", "结果", "分析", "消融", "数据集"],
  critic: ["abstract", "introduction", "related work", "discussion", "limitation", "conclusion", "future", "摘要", "引言", "相关工作", "讨论", "局限", "结论", "未来"],
};

function chapterAgentRoles(chapter) {
  const raw = `${chapter?.title || ""} ${chapter?.display_title || ""} ${chapter?.displayTitle || ""}`.toLowerCase();
  const roles = Object.entries(chapterAgentKeywords)
    .filter(([, keywords]) => keywords.some((keyword) => raw.includes(keyword.toLowerCase())))
    .map(([agent]) => agent);
  return roles.length ? roles : ["method", "experiment", "critic"];
}

function chapterStatus(chapter, { selectedFile, hasParsedSections, isAnalyzing, workflowFinished, analysisError, agentStates }) {
  if (selectedFile && !hasParsedSections) {
    return { label: "待识别", tone: "pending", icon: "dot" };
  }
  if (analysisError) {
    return { label: "已识别", tone: "ready", icon: "dot" };
  }
  if (workflowFinished) {
    return { label: "已纳入", tone: "done", icon: "check" };
  }
  if (isAnalyzing) {
    const roles = chapterAgentRoles(chapter);
    const isActive = roles.some((role) => agentStates[role] === "running");
    return isActive
      ? { label: "分析中", tone: "running", icon: "pulse" }
      : { label: "已识别", tone: "ready", icon: "dot" };
  }
  return { label: selectedFile ? "已识别" : "示例", tone: "ready", icon: "dot" };
}

function pendingAnalysisForFile(file) {
  return {
    mode: "pending",
    paper: {
      title: file.name.replace(/\.pdf$/i, "") || "Uploaded Paper",
      filename: file.name,
      pages: "Pending",
      sections_count: "Pending",
      size_bytes: file.size,
      sections: [],
    },
  };
}

function markdownFromAnalysis(data) {
  const summary = data.summary_output;
  const novelty = data.assessment?.novelty;
  const reliability = data.assessment?.reliability;
  if (!summary) {
    return `# ${data.paper?.title || "Paper Reader Notes"}\n\nAnalysis is still running.`;
  }
  return `# ${data.paper.title}

## 一句话总结
${summary.one_sentence_summary}

## 核心贡献
${summary.core_contributions.map((item) => `- ${item}`).join("\n")}

## 方法要点
${summary.method_highlights}

## 实验要点
${summary.experiment_highlights}

## 局限与未来工作
${summary.limitations_and_future_work}

## 评估
- 创新性：${novelty ? `${novelty.score} / 5（${novelty.label}）` : "未评估"}
- 分析可靠度：${reliability ? `${reliability.score} / 100（${reliability.label}）` : "未评估"}
${reliability?.warnings?.length ? `- 注意：${reliability.warnings.join("；")}` : ""}

## 研读笔记
${summary.reading_notes || ""}
`;
}

function Section({ title, children }) {
  return (
    <section className="result-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function NumberedList({ items }) {
  return (
    <ol className="numbered-list">
      {(items || []).map((item, index) => (
        <li key={`${item}-${index}`}>
          <span>{index + 1}</span>
          <p>{item}</p>
        </li>
      ))}
    </ol>
  );
}

function BulletList({ items }) {
  return (
    <ul className="bullets">
      {(items || []).map((item, index) => (
        <li key={`${item}-${index}`}>{item}</li>
      ))}
    </ul>
  );
}

function TagList({ items }) {
  return (
    <div className="tag-list">
      {(items || []).map((item) => (
        <span key={item}>{item}</span>
      ))}
    </div>
  );
}

const noveltyDimensionLabels = {
  problem_originality: "问题定义",
  method_originality: "方法机制",
  prior_work_difference: "已有工作差异",
  generality: "适用范围",
};

const reliabilityBreakdownLabels = {
  parsing: ["解析", 20],
  coverage: ["覆盖", 35],
  citations: ["引用", 30],
  output_integrity: ["输出", 15],
};

function AssessmentDetails({ summary, rows, warnings = [] }) {
  if (!rows.length && !warnings.length) return null;

  return (
    <details className="assessment-details">
      <summary>{summary}</summary>
      <div className="assessment-breakdown">
        {rows.map(({ label, value, detail }) => (
          <div className="assessment-row" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
            {detail && <p>{detail}</p>}
          </div>
        ))}
        {warnings.map((warning) => (
          <p className="assessment-warning" key={warning}>{warning}</p>
        ))}
      </div>
    </details>
  );
}

function ScoreGrid({ critic, assessment }) {
  const novelty = assessment?.novelty;
  const reliability = assessment?.reliability;
  const noveltyScore = novelty?.score ?? critic.novelty_score ?? "—";
  const noveltyLabel = novelty?.label || (critic.novelty_score >= 4 ? "创新性较高" : "评审估计");
  const dimensions = novelty?.dimensions || critic.novelty_dimensions || [];
  const noveltyRows = dimensions.map((dimension) => ({
    label: noveltyDimensionLabels[dimension.dimension] || dimension.dimension,
    value: `${dimension.score} / 5`,
    detail: `${dimension.reason}${dimension.evidence_ids?.length ? `（证据：${dimension.evidence_ids.join("、")}）` : ""}`,
  }));
  const reliabilityRows = Object.entries(reliability?.breakdown || {}).map(([key, value]) => ({
    label: reliabilityBreakdownLabels[key]?.[0] || key,
    value: `${value} / ${reliabilityBreakdownLabels[key]?.[1] || "—"}`,
  }));
  if (reliability && reliability.raw_score !== reliability.score) {
    reliabilityRows.push(
      { label: "分项原始分", value: `${reliability.raw_score} / 100` },
      { label: "证据条件上限", value: `${reliability.score_cap} / 100` },
    );
  }

  return (
    <div className="score-grid">
      <div>
        <h3>创新性评分</h3>
        <div className="score-content">
          <strong>
            {noveltyScore} <span>/ 5</span>
          </strong>
          <small>{noveltyLabel}</small>
          <p>{critic.novelty_justification || "模型未返回创新性评分依据。"}</p>
          <AssessmentDetails
            summary="查看评分依据"
            rows={noveltyRows}
            warnings={novelty?.warnings}
          />
        </div>
      </div>
      <div>
        <h3>分析可靠度</h3>
        <div className="score-content">
          <strong className={`confidence ${reliability?.level || "unknown"}`}>
            {reliability ? reliability.score : "—"} <span>/ 100</span>
          </strong>
          <small className={`reliability-label ${reliability?.level || "unknown"}`}>
            {reliability?.label || "尚未评估"}
          </small>
          <p>基于解析完整度、关键章节覆盖、证据引用与输出完整性计算。</p>
          <AssessmentDetails
            summary="查看可靠度构成"
            rows={reliabilityRows}
            warnings={reliability?.warnings}
          />
        </div>
      </div>
    </div>
  );
}

function StreamPlaceholder({ title, message }) {
  return (
    <section className="stream-placeholder">
      <span className="pulse-dot" />
      <div>
        <h3>{title}</h3>
        <p>{message || "正在等待后端流式结果..."}</p>
      </div>
    </section>
  );
}

const streamAgentNames = {
  method: "方法 Agent",
  experiment: "实验 Agent",
  critic: "评审 Agent",
  summary: "总结 Agent",
};

function TokenStreamPreview({ streams }) {
  const activeStreams = Object.entries(streams || {}).filter(([, text]) => text?.trim());
  if (!activeStreams.length) return null;

  return (
    <section className="token-preview">
      <div className="token-preview-heading">
        <span className="pulse-dot" />
        <strong>实时生成</strong>
        <small>正在接收模型 token，完成后会自动整理为结构化结果</small>
      </div>
      <div className="token-streams">
        {activeStreams.map(([agent, text]) => (
          <article key={agent}>
            <span>{streamAgentNames[agent] || agent}</span>
            <pre>{text.slice(-700)}</pre>
          </article>
        ))}
      </div>
    </section>
  );
}

function EvidenceList({ items }) {
  if (!items?.length) return null;
  return (
    <Section title="证据依据">
      <div className="evidence-list">
        {items.slice(0, 5).map((item, index) => (
          <article key={`${item.id || index}-${item.quote || index}`}>
            <div>
              <strong>{item.id || `E${index + 1}`}</strong>
              <span>{item.section || "原文片段"} · {item.page || "页码待确认"}</span>
            </div>
            <p>{item.quote}</p>
            {item.note && <small>{item.note}</small>}
          </article>
        ))}
      </div>
    </Section>
  );
}

function ResultContent({ activeTab, data, error, streamMessage, agentStreams, isAnalyzing }) {
  const showLivePreview = isAnalyzing && !error;

  if (error) {
    return (
      <>
        <Section title="分析需要处理">
          <div className="error-callout">
            <IconAlertCircle size={20} />
            <p>{error}</p>
          </div>
        </Section>
        <Section title="当前已连通的功能">
          <BulletList
            items={[
              "React 前端已能将上传的 PDF 发送到 Python 后端。",
              "前端通过 /api/analyze/stream 接收流式分析结果。",
              "配置 GLM_API_KEY 后即可运行真实分析。",
              "Demo 模式可以在不使用模型凭证的情况下验证上传与解析链路。",
            ]}
          />
        </Section>
      </>
    );
  }

  const { method_output: method, experiment_output: experiment, critic_output: critic, summary_output: summary } = data;

  if (activeTab === "方法") {
    if (!method) {
      return <StreamPlaceholder title="方法分析 Agent 正在工作" message={streamMessage} />;
    }
    return (
      <>
        {showLivePreview && <TokenStreamPreview streams={agentStreams} />}
        <Section title="研究问题"><p>{method.research_problem}</p></Section>
        <Section title="提出的方法"><p>{method.proposed_method}</p></Section>
        <Section title="关键组件"><TagList items={method.key_components} /></Section>
        <Section title="创新点"><BulletList items={method.innovations} /></Section>
        <Section title="与已有工作的差异"><p>{method.differences_from_prior}</p></Section>
        {method.implementation_details && <Section title="实现细节"><p>{method.implementation_details}</p></Section>}
      </>
    );
  }

  if (activeTab === "实验") {
    if (!experiment) {
      return <StreamPlaceholder title="实验分析 Agent 正在工作" message={streamMessage} />;
    }
    return (
      <>
        {showLivePreview && <TokenStreamPreview streams={agentStreams} />}
        <Section title="数据集"><TagList items={experiment.datasets} /></Section>
        <Section title="评估指标"><TagList items={experiment.metrics} /></Section>
        <Section title="主要结果"><p>{experiment.main_results}</p></Section>
        <Section title="基线对比"><p>{experiment.comparison_with_baselines}</p></Section>
        {experiment.ablation_study && <Section title="消融实验"><p>{experiment.ablation_study}</p></Section>}
        <Section title="重要发现"><BulletList items={experiment.notable_findings} /></Section>
      </>
    );
  }

  if (activeTab === "批判性评审") {
    if (!critic) {
      return <StreamPlaceholder title="批判性评审 Agent 正在工作" message={streamMessage} />;
    }
    return (
      <>
        {showLivePreview && <TokenStreamPreview streams={agentStreams} />}
        <Section title="创新性评分依据"><p>{critic.novelty_justification}</p></Section>
        <Section title="优点"><BulletList items={critic.strengths} /></Section>
        <Section title="局限"><BulletList items={critic.limitations} /></Section>
        <Section title="改进方向"><BulletList items={critic.potential_improvements} /></Section>
        {critic.broader_impact && <Section title="潜在影响"><p>{critic.broader_impact}</p></Section>}
      </>
    );
  }

  if (activeTab === "最终笔记") {
    if (!summary) {
      return <StreamPlaceholder title="总结 Agent 正在等待" message={streamMessage} />;
    }
    return (
      <>
        {showLivePreview && <TokenStreamPreview streams={agentStreams} />}
        <Section title="精炼研读笔记"><p>{summary.reading_notes || summary.one_sentence_summary}</p></Section>
        <Section title="方法要点"><p>{summary.method_highlights}</p></Section>
        <Section title="实验要点"><p>{summary.experiment_highlights}</p></Section>
        <Section title="局限与未来工作"><p>{summary.limitations_and_future_work}</p></Section>
      </>
    );
  }

  if (!summary || !critic) {
    return (
      <>
        {showLivePreview && <TokenStreamPreview streams={agentStreams} />}
        <StreamPlaceholder title="正在接收流式分析" message={streamMessage} />
        {method && <Section title="方法结果预览"><p>{method.proposed_method}</p></Section>}
        {experiment && <Section title="实验结果预览"><p>{experiment.main_results}</p></Section>}
        {critic && <Section title="评审结果预览"><p>{critic.novelty_justification}</p></Section>}
      </>
    );
  }

  return (
    <>
      {showLivePreview && <TokenStreamPreview streams={agentStreams} />}
      <Section title="一句话总结"><p>{summary.one_sentence_summary}</p></Section>
      <Section title="核心贡献"><NumberedList items={summary.core_contributions} /></Section>
      <ScoreGrid critic={critic} assessment={data.assessment} />
      <Section title="关键发现">
        <BulletList
          items={[
            summary.method_highlights,
            summary.experiment_highlights,
            summary.limitations_and_future_work,
          ].filter(Boolean)}
        />
      </Section>
    </>
  );
}

function AppButton({ children, className = "", ...props }) {
  return (
    <button className={`button ${className}`} type="button" {...props}>
      {children}
    </button>
  );
}

const chatContextKeys = [
  "mode",
  "paper",
  "method_output",
  "experiment_output",
  "critic_output",
  "summary_output",
  "assessment",
  "evidence_index",
];

function analysisContextForChat(data) {
  return chatContextKeys.reduce((context, key) => {
    if (data?.[key] !== undefined) context[key] = data[key];
    return context;
  }, {});
}

function PaperChatDrawer({
  paperTitle,
  modelLabel = "论文研究助手",
  conversations,
  activeConversationId,
  messages,
  input,
  quote,
  isStreaming,
  isConversationLoading,
  onInputChange,
  onClearQuote,
  onSend,
  onClose,
  onConversationChange,
  onNewConversation,
  onDeleteConversation,
  onRenameConversation,
}) {
  const textareaRef = useRef(null);
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const {
    width: drawerWidth,
    minWidth: drawerMinWidth,
    maxWidth: drawerMaxWidth,
    isResizing,
    drawerStyle,
    startResize,
    handleResizeKeyDown,
  } = useResizableChatDrawer();
  const activeConversation = conversations.find((conversation) => conversation.id === activeConversationId);
  const {
    containerRef: messagesContainerRef,
    autoFollow,
    handleScroll,
    scrollToBottom,
  } = useChatAutoScroll(messages, isStreaming, activeConversationId || "new");

  useEffect(() => {
    textareaRef.current?.focus();
  }, [quote]);

  useEffect(() => {
    setIsRenaming(false);
    setRenameValue(activeConversation?.title || "");
  }, [activeConversationId, activeConversation?.title]);

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (input.trim() && !isStreaming) onSend();
    }
  }

  async function submitRename() {
    const title = renameValue.trim();
    if (!activeConversationId || !title) return;
    const saved = await onRenameConversation(title);
    if (saved !== false) setIsRenaming(false);
  }

  function handleRenameKeyDown(event) {
    if (event.key === "Enter" && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submitRename();
    }
    if (event.key === "Escape") {
      setRenameValue(activeConversation?.title || "");
      setIsRenaming(false);
    }
  }

  return (
    <section
      className={`paper-chat-drawer${isResizing ? " resizing" : ""}`}
      id="paper-chat-drawer"
      aria-label="论文追问"
      style={drawerStyle}
    >
      <div
        className="chat-resize-handle"
        role="separator"
        aria-label="调整论文追问宽度"
        aria-orientation="vertical"
        aria-valuemin={drawerMinWidth}
        aria-valuemax={drawerMaxWidth}
        aria-valuenow={Math.round(drawerWidth)}
        tabIndex={0}
        title="拖动调整论文追问宽度"
        onPointerDown={startResize}
        onKeyDown={handleResizeKeyDown}
      >
        <IconGripVertical size={16} stroke={1.8} />
      </div>
      <header className="chat-header">
        <div>
          <span><IconMessageCircle size={16} stroke={1.8} /> 论文追问</span>
          <strong title={paperTitle}>{paperTitle}</strong>
        </div>
        <div className="chat-header-actions">
          <button
            type="button"
            title="删除当前对话"
            aria-label="删除当前对话"
            onClick={onDeleteConversation}
            disabled={!activeConversationId || isStreaming || isConversationLoading}
          >
            <IconTrash size={17} stroke={1.8} />
          </button>
          <button type="button" title="关闭追问" aria-label="关闭追问" onClick={onClose}>
            <IconX size={18} stroke={1.8} />
          </button>
        </div>
      </header>

      <div className="chat-session-bar">
        {isRenaming ? (
          <>
            <input
              type="text"
              aria-label="修改会话名称"
              value={renameValue}
              maxLength={80}
              autoFocus
              onChange={(event) => setRenameValue(event.target.value)}
              onKeyDown={handleRenameKeyDown}
              disabled={isConversationLoading}
            />
            <button
              type="button"
              title="保存名称"
              aria-label="保存会话名称"
              onClick={() => void submitRename()}
              disabled={!renameValue.trim() || isConversationLoading}
            >
              <IconCheck size={17} stroke={2} />
            </button>
            <button
              type="button"
              title="取消编辑"
              aria-label="取消修改会话名称"
              onClick={() => {
                setRenameValue(activeConversation?.title || "");
                setIsRenaming(false);
              }}
              disabled={isConversationLoading}
            >
              <IconX size={17} stroke={2} />
            </button>
          </>
        ) : (
          <>
            <select
              aria-label="选择论文追问会话"
              value={activeConversationId || ""}
              onChange={(event) => onConversationChange(event.target.value)}
              disabled={isStreaming || isConversationLoading}
            >
              <option value="">新对话</option>
              {conversations.map((conversation) => (
                <option value={conversation.id} key={conversation.id}>
                  {conversation.title}（{conversation.message_count}）
                </option>
              ))}
            </select>
            <button
              type="button"
              title="修改当前会话名称"
              aria-label="修改当前会话名称"
              onClick={() => {
                setRenameValue(activeConversation?.title || "");
                setIsRenaming(true);
              }}
              disabled={!activeConversationId || isStreaming || isConversationLoading}
            >
              <IconPencil size={16} stroke={1.9} />
            </button>
            <button
              type="button"
              title="新建对话"
              aria-label="新建对话"
              onClick={onNewConversation}
              disabled={isStreaming || isConversationLoading}
            >
              <IconPlus size={17} stroke={2} />
            </button>
          </>
        )}
      </div>

      <div className="chat-messages" ref={messagesContainerRef} onScroll={handleScroll}>
        {isConversationLoading && (
          <div className="chat-empty">
            <IconLoader2 className="spin" size={22} stroke={1.5} />
            <strong>正在恢复对话</strong>
          </div>
        )}
        {!isConversationLoading && !messages.length && (
          <div className="chat-empty">
            <IconMessageCircle size={24} stroke={1.5} />
            <strong>{modelLabel}</strong>
          </div>
        )}
        {messages.map((message) => (
          <article className={`chat-message ${message.role}${message.error ? " error" : ""}`} key={message.id}>
            {message.quote && (
              <blockquote><IconQuote size={14} stroke={1.8} /> {message.quote}</blockquote>
            )}
            {message.content ? (
              message.role === "assistant" ? (
                <Suspense fallback={<p>{message.content}</p>}>
                  <ChatMarkdown>{message.content}</ChatMarkdown>
                </Suspense>
              ) : (
                <p>{message.content}</p>
              )
            ) : (
              <span className="chat-typing" aria-label="正在生成"><i /><i /><i /></span>
            )}
            {message.role === "assistant" && <ModelCallTrace trace={message.model_trace} />}
          </article>
        ))}
      </div>

      {!autoFollow && (
        <button
          className="chat-scroll-to-bottom"
          type="button"
          title="回到最新回答"
          aria-label="回到最新回答"
          onClick={() => scrollToBottom("smooth")}
        >
          <IconArrowDown size={17} stroke={2} />
        </button>
      )}

      <form className="chat-composer" onSubmit={(event) => { event.preventDefault(); onSend(); }}>
        {quote && (
          <div className="chat-quote-chip">
            <IconQuote size={14} stroke={1.8} />
            <span>{quote}</span>
            <button type="button" aria-label="移除引用片段" onClick={onClearQuote}><IconX size={14} /></button>
          </div>
        )}
        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            value={input}
            maxLength={4000}
            rows={2}
            placeholder="继续追问这篇论文"
            onChange={(event) => onInputChange(event.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button className="chat-send" type="submit" aria-label="发送问题" disabled={!input.trim() || isStreaming}>
            {isStreaming ? <IconLoader2 className="spin" size={18} /> : <IconSend size={18} stroke={1.9} />}
          </button>
        </div>
      </form>
    </section>
  );
}

function SettingsDialog({
  status,
  loading,
  error,
  modelHealth,
  healthLoading,
  healthError,
  routing,
  credentialProvider,
  baseUrl,
  apiKey,
  apiKeyVisible,
  saving,
  feedback,
  onRoutingChange,
  onSaveRouting,
  onCredentialProviderChange,
  onBaseUrlChange,
  onApiKeyChange,
  onToggleApiKey,
  onSaveApiKey,
  onRefreshHealth,
  onRetry,
  onClose,
}) {
  const providers = status?.providers || [];
  const textProvider = providers.find((provider) => provider.id === routing.text_provider) || providers[0];
  const visionProvider = textProvider;
  const credential = providers.find((provider) => provider.id === credentialProvider) || providers[0];
  const configuredProviderCount = providers.filter((provider) => provider.configured).length;

  return (
    <div
      className="settings-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !saving) onClose();
      }}
    >
      <section
        className="settings-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-dialog-title"
      >
        <header className="settings-dialog-header">
          <div>
            <span className="settings-title-icon"><IconSettings size={19} stroke={1.8} /></span>
            <div>
              <strong id="settings-dialog-title">Settings</strong>
              <small>应用配置</small>
            </div>
          </div>
          <button type="button" title="关闭设置" aria-label="关闭设置" onClick={onClose} disabled={saving}>
            <IconX size={19} stroke={1.8} />
          </button>
        </header>

        {loading && (
          <div className="settings-loading">
            <IconLoader2 className="spin" size={22} />
            <span>正在读取配置</span>
          </div>
        )}

        {!loading && error && (
          <div className="settings-load-error">
            <IconAlertCircle size={19} />
            <span>{error}</span>
            <button type="button" onClick={onRetry}>重新加载</button>
          </div>
        )}

        {!loading && !error && status && (
          <div className="settings-content">
            <div className="settings-meta-row">
              <div>
                <small>项目版本</small>
                <strong>{status.version || "V1.2.1"}</strong>
              </div>
              <div>
                <small>模型服务</small>
                <span className={configuredProviderCount ? "configured" : "unconfigured"}>
                  <i /> {configuredProviderCount} / {providers.length} 个厂商已配置
                </span>
              </div>
            </div>

            <section className="settings-section model-health-section">
              <header>
                <div>
                  <h3>模型可用性</h3>
                  <small>
                    {modelHealth?.checked_at
                      ? `自动验证于 ${new Date(modelHealth.checked_at).toLocaleString()}`
                      : "打开 Settings 时自动验证已配置厂商；视觉模型会发起最小真实请求"}
                  </small>
                </div>
                <button
                  className="settings-secondary-action"
                  type="button"
                  onClick={onRefreshHealth}
                  disabled={healthLoading}
                >
                  <IconRefresh className={healthLoading ? "spin" : ""} size={15} />
                  {healthLoading ? "验证中" : "立即验证"}
                </button>
              </header>
              {healthError && (
                <div className="settings-health-error" role="status">
                  <IconAlertCircle size={15} />
                  <span>{healthError}</span>
                </div>
              )}
              <div className="settings-health-grid" aria-busy={healthLoading}>
                {(modelHealth?.providers || providers.map((provider) => ({
                  id: provider.id,
                  label: provider.label,
                  status: provider.configured ? "checking" : "unconfigured",
                  message: provider.configured ? "等待真实验证。" : "未配置 API Key。",
                }))).map((provider) => {
                  const statusLabels = {
                    ok: "正常",
                    drift: "目录变化",
                    unavailable: "不可用",
                    unconfigured: "未配置",
                    checking: "等待检查",
                  };
                  const missingModels = [
                    ...(provider.missing_text_models || []),
                    ...(provider.missing_vision_models || []),
                  ];
                  return (
                    <article className={`settings-health-card ${provider.status}`} key={provider.id}>
                      <div>
                        <strong>{provider.label}</strong>
                        <span><i /> {statusLabels[provider.status] || provider.status}</span>
                      </div>
                      <small>{provider.message}</small>
                      {missingModels.length > 0 && (
                        <code title={missingModels.join(", ")}>缺失：{missingModels.join(", ")}</code>
                      )}
                    </article>
                  );
                })}
              </div>
            </section>

            <section className="settings-section model-routing-section">
              <header>
                <div>
                  <h3>模型路由</h3>
                  <small>视觉理解自动跟随文本厂商</small>
                </div>
                <button
                  className="settings-secondary-action"
                  type="button"
                  onClick={onSaveRouting}
                  disabled={saving || !routing.text_model}
                >
                  {saving && feedback?.scope === "routing" ? <IconLoader2 className="spin" size={15} /> : <IconCheck size={15} />}
                  应用配置
                </button>
              </header>
              <div className="settings-route-list">
                <div className="settings-route-row">
                  <span className="model-kind text"><IconCpu size={18} /></span>
                  <div className="settings-route-copy">
                    <strong>文本分析</strong>
                    <small>Agent 分析、总结与论文追问</small>
                  </div>
                  <label>
                    <span>厂商</span>
                    <select
                      value={routing.text_provider}
                      onChange={(event) => onRoutingChange("text_provider", event.target.value)}
                      disabled={saving}
                    >
                      {providers.map((provider) => <option value={provider.id} key={provider.id}>{provider.label}</option>)}
                    </select>
                  </label>
                  <label>
                    <span>模型</span>
                    <select
                      value={routing.text_model}
                      onChange={(event) => onRoutingChange("text_model", event.target.value)}
                      disabled={saving}
                    >
                      {(textProvider?.text_models || []).map((model) => (
                        <option value={model.id} key={model.id}>{model.label}</option>
                      ))}
                    </select>
                  </label>
                  <span className={`model-state ${textProvider?.configured ? "configured" : "unconfigured"}`}>
                    <i /> {textProvider?.configured ? "Key 已配置" : "缺少 Key"}
                  </span>
                </div>

                <div className={`settings-route-row vision-route ${routing.vision_enabled ? "" : "disabled"}`}>
                  <span className="model-kind vision"><IconPhoto size={18} /></span>
                  <div className="settings-route-copy">
                    <strong>图表理解</strong>
                    <small>{visionProvider?.supports_vision ? "论文图像、图表与公式区域" : "官方云 API 暂不支持图像输入"}</small>
                  </div>
                  <label>
                    <span>厂商</span>
                    <div className="settings-locked-field">
                      <strong>{visionProvider?.label || "未选择"}</strong>
                      <small>跟随文本</small>
                    </div>
                  </label>
                  <label>
                    <span>模型</span>
                    <div className="settings-locked-field model">
                      <strong>{routing.vision_model || "官方云 API 暂不支持"}</strong>
                      <small>{routing.vision_model ? "自动配对" : "文本可用"}</small>
                    </div>
                  </label>
                  <label className="settings-toggle">
                    <input
                      type="checkbox"
                      checked={routing.vision_enabled}
                      onChange={(event) => onRoutingChange("vision_enabled", event.target.checked)}
                      disabled={saving || !visionProvider?.supports_vision}
                    />
                    <span aria-hidden="true" />
                    启用
                  </label>
                </div>
              </div>
              {feedback?.scope === "routing" && feedback.message && (
                <div className={`settings-feedback ${feedback.tone}`} role="status">
                  {feedback.tone === "success" ? <IconCheck size={16} /> : <IconAlertCircle size={16} />}
                  <span>{feedback.message}</span>
                </div>
              )}
            </section>

            <section className="settings-section api-key-section">
              <header>
                <div>
                  <h3>厂商凭据</h3>
                  <small>密钥按厂商分别保存在本机</small>
                </div>
                <a href={credential?.key_url} target="_blank" rel="noreferrer">
                  获取 Key <IconExternalLink size={13} />
                </a>
              </header>
              <form onSubmit={onSaveApiKey}>
                <div className="settings-credential-grid">
                  <label>
                    <span>模型厂商</span>
                    <select
                      value={credentialProvider}
                      onChange={(event) => onCredentialProviderChange(event.target.value)}
                      disabled={saving}
                    >
                      {providers.map((provider) => (
                        <option value={provider.id} key={provider.id}>
                          {provider.label}{provider.configured ? " · 已配置" : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <span>Base URL</span>
                    <input
                      type="url"
                      value={baseUrl}
                      onChange={(event) => onBaseUrlChange(event.target.value)}
                      disabled={saving}
                      spellCheck="false"
                    />
                  </label>
                </div>
                <label htmlFor="settings-api-key">API Key</label>
                <div className="settings-key-input">
                  <IconKey size={17} stroke={1.8} />
                  <input
                    id="settings-api-key"
                    type={apiKeyVisible ? "text" : "password"}
                    value={apiKey}
                    autoComplete="off"
                    spellCheck="false"
                    placeholder={credential?.configured ? `粘贴新的 Key 以替换 ${credential.label} 配置` : `粘贴 ${credential?.label || "厂商"} API Key`}
                    onChange={(event) => onApiKeyChange(event.target.value)}
                    disabled={saving}
                  />
                  <button
                    type="button"
                    title={apiKeyVisible ? "隐藏 API Key" : "显示 API Key"}
                    aria-label={apiKeyVisible ? "隐藏 API Key" : "显示 API Key"}
                    onClick={onToggleApiKey}
                    disabled={!apiKey || saving}
                  >
                    {apiKeyVisible ? <IconEyeOff size={17} /> : <IconEye size={17} />}
                  </button>
                </div>
                <div className="settings-key-footer">
                  <small>仅写入本机 <code>.env</code>，保存后不会回显。</small>
                  <button type="submit" disabled={apiKey.trim().length < 10 || saving}>
                    {saving && feedback?.scope === "credential" ? <IconLoader2 className="spin" size={16} /> : <IconShieldCheck size={16} />}
                    {saving && feedback?.scope === "credential" ? "正在验证" : "验证并保存"}
                  </button>
                </div>
                {feedback?.scope === "credential" && feedback.message && (
                  <div className={`settings-feedback ${feedback.tone}`} role="status">
                    {feedback.tone === "success" ? <IconCheck size={16} /> : <IconAlertCircle size={16} />}
                    <span>{feedback.message}</span>
                  </div>
                )}
              </form>
            </section>
          </div>
        )}
      </section>
    </div>
  );
}

export function App() {
  const [workspaceMode, setWorkspaceMode] = useState("reading");
  const [workspaceMenuOpen, setWorkspaceMenuOpen] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [activeTab, setActiveTab] = useState("概览");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsStatus, setSettingsStatus] = useState(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [settingsError, setSettingsError] = useState("");
  const [settingsModelHealth, setSettingsModelHealth] = useState(null);
  const [settingsHealthLoading, setSettingsHealthLoading] = useState(false);
  const [settingsHealthError, setSettingsHealthError] = useState("");
  const [settingsRouting, setSettingsRouting] = useState(defaultSettingsRouting);
  const [settingsCredentialProvider, setSettingsCredentialProvider] = useState("zhipu");
  const [settingsBaseUrl, setSettingsBaseUrl] = useState("");
  const [settingsApiKey, setSettingsApiKey] = useState("");
  const [settingsApiKeyVisible, setSettingsApiKeyVisible] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsFeedback, setSettingsFeedback] = useState(null);
  const [chaptersOpen, setChaptersOpen] = useState(true);
  const [recentOpen, setRecentOpen] = useState(false);
  const [toast, setToast] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedChapterIndex, setSelectedChapterIndex] = useState(0);
  const [analysisData, setAnalysisData] = useState(sampleAnalysis);
  const [analysisError, setAnalysisError] = useState("");
  const [agentStates, setAgentStates] = useState(emptyAgentStates);
  const [agentStreams, setAgentStreams] = useState(emptyAgentStreams);
  const [streamMessage, setStreamMessage] = useState("已准备好开始分析");
  const [selectionAction, setSelectionAction] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatQuote, setChatQuote] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatStreaming, setChatStreaming] = useState(false);
  const [chatConversations, setChatConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState("");
  const [chatConversationLoading, setChatConversationLoading] = useState(false);
  const [historyItems, setHistoryItems] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");
  const [historyBusyId, setHistoryBusyId] = useState("");
  const fileInputRef = useRef(null);
  const resultsPanelRef = useRef(null);
  const resultsScrollRef = useRef(null);
  const chatAbortRef = useRef(null);
  const workspaceMenuRef = useRef(null);

  useEffect(() => {
    void loadPaperHistory();
    void loadApplicationSettings();
    return () => chatAbortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (!settingsOpen) return undefined;
    function closeSettingsWithKeyboard(event) {
      if (event.key === "Escape" && !settingsSaving) setSettingsOpen(false);
    }
    document.addEventListener("keydown", closeSettingsWithKeyboard);
    return () => document.removeEventListener("keydown", closeSettingsWithKeyboard);
  }, [settingsOpen, settingsSaving]);

  useEffect(() => {
    function closeWorkspaceMenu(event) {
      if (!workspaceMenuRef.current?.contains(event.target)) setWorkspaceMenuOpen(false);
    }
    function closeWorkspaceMenuWithKeyboard(event) {
      if (event.key === "Escape") setWorkspaceMenuOpen(false);
    }
    document.addEventListener("pointerdown", closeWorkspaceMenu);
    document.addEventListener("keydown", closeWorkspaceMenuWithKeyboard);
    return () => {
      document.removeEventListener("pointerdown", closeWorkspaceMenu);
      document.removeEventListener("keydown", closeWorkspaceMenuWithKeyboard);
    };
  }, []);

  const displayedData = analysisData || sampleAnalysis;
  const displayedPaper = displayedData.paper;
  const activeModelLabel = settingsStatus?.routing?.text?.model_label
    || settingsStatus?.routing?.text?.model
    || "论文研究助手";
  const hasFinalAnalysis = Boolean(
    displayedData.method_output &&
      displayedData.experiment_output &&
      displayedData.critic_output &&
      displayedData.summary_output,
  );
  const hasExportData = workspaceMode === "comparison"
    ? Boolean(comparisonData?.comparison)
    : hasFinalAnalysis;
  const sourceSections = displayedPaper.sections?.length
    ? displayedPaper.sections
    : selectedFile
      ? [{ title: "等待解析", display_title: "等待解析", chars: 0 }]
      : sampleAnalysis.paper.sections;
  const agents = useMemo(
    () =>
      agentBase.map((agent) => {
        const state = agentStates[agent.id] || "waiting";
        const complete = state === "complete" && !analysisError;
        const active = state === "running";
        const failed = state === "failed" || Boolean(analysisError && state === "running");
        return {
          ...agent,
          state,
          complete,
          active,
          failed,
          progress: complete ? 100 : active ? 34 : failed ? 18 : 0,
          status: failed ? "需要处理" : complete ? "已完成" : active ? "分析中" : "未开始",
          statusTone: failed ? "failed" : complete ? "complete" : active ? "active" : "waiting",
          steps: agentStepLabels.map((label, index) => ({
            label,
            state: complete ? "done" : failed && index === 0 ? "failed" : active && index === 0 ? "active" : "pending",
          })),
        };
      }),
    [agentStates, analysisError],
  );

  const workflowFinished = Boolean(
    hasFinalAnalysis &&
      !analysisError &&
      !isAnalyzing &&
      (displayedData.mode === "live" || displayedData.mode === "demo"),
  );
  const workflowLabel = isAnalyzing
    ? streamMessage
    : analysisError
      ? "分析需要处理"
      : workflowFinished
        ? "所有 Agent 已完成"
        : selectedFile
          ? "等待开始分析"
          : "等待上传论文";
  const workflowDetail = isAnalyzing
    ? "正在接收后端事件"
    : workflowFinished
      ? displayedData.mode === "live" ? "真实后端结果" : "Demo 结果"
      : "应用已连接";
  const workflowDotClass = isAnalyzing
    ? "pulse-dot"
    : analysisError
      ? "error-dot"
      : workflowFinished
        ? "complete-dot"
        : "idle-dot";
  const hasParsedSections = Boolean(displayedPaper.sections?.length);
  const displaySections = sourceSections.map((chapter, index) => ({
        ...chapter,
        displayTitle: cleanChapterTitle(chapter, index),
        meta: selectedFile && !hasParsedSections ? "点击 Analyze Paper 后自动识别章节" : chapterMeta(chapter),
        status: chapterStatus(chapter, {
          selectedFile,
          hasParsedSections,
          isAnalyzing,
          workflowFinished,
          analysisError,
          agentStates,
        }),
      }));

  function showToast(message) {
    setToast(message);
    window.setTimeout(() => setToast(""), 2200);
  }

  async function loadApplicationSettings() {
    setSettingsLoading(true);
    setSettingsError("");
    try {
      const response = await fetch("/api/settings");
      if (!response.ok) throw new Error(`Settings request failed（HTTP ${response.status}）`);
      const payload = await response.json();
      const textRoute = payload.routing?.text;
      const visionRoute = payload.routing?.vision;
      const initialCredentialProvider = textRoute?.provider || payload.providers?.[0]?.id || "zhipu";
      const initialCredential = payload.providers?.find((provider) => provider.id === initialCredentialProvider);
      setSettingsStatus(payload);
      setSettingsRouting({
        text_provider: textRoute?.provider || "zhipu",
        text_model: textRoute?.model || "glm-5.2",
        vision_enabled: Boolean(visionRoute?.enabled && initialCredential?.supports_vision),
        vision_provider: textRoute?.provider || "zhipu",
        vision_model: initialCredential?.default_vision_model || "",
      });
      setSettingsCredentialProvider(initialCredentialProvider);
      setSettingsBaseUrl(initialCredential?.base_url || "");
    } catch (error) {
      setSettingsError(error instanceof Error ? error.message : "无法读取应用配置。");
    } finally {
      setSettingsLoading(false);
    }
  }

  async function loadModelHealth(force = false) {
    if (settingsHealthLoading) return;
    setSettingsHealthLoading(true);
    setSettingsHealthError("");
    try {
      const suffix = force ? "?force=true" : "";
      const response = await fetch(`/api/settings/model-health${suffix}`);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `模型可用性验证失败（HTTP ${response.status}）`);
      setSettingsModelHealth(payload);
    } catch (error) {
      setSettingsHealthError(error instanceof Error ? error.message : "模型可用性验证失败。");
    } finally {
      setSettingsHealthLoading(false);
    }
  }

  function openApplicationSettings() {
    setHistoryOpen(false);
    setWorkspaceMenuOpen(false);
    setSettingsApiKey("");
    setSettingsBaseUrl("");
    setSettingsApiKeyVisible(false);
    setSettingsFeedback(null);
    setSettingsOpen(true);
    void loadApplicationSettings();
    void loadModelHealth();
  }

  function closeApplicationSettings() {
    if (settingsSaving) return;
    setSettingsOpen(false);
    setSettingsApiKey("");
    setSettingsApiKeyVisible(false);
    setSettingsFeedback(null);
  }

  function changeSettingsRouting(field, value) {
    if (field === "text_provider") {
      const provider = settingsStatus?.providers?.find((item) => item.id === value);
      setSettingsCredentialProvider(value);
      setSettingsBaseUrl(provider?.base_url || "");
      setSettingsApiKey("");
      setSettingsApiKeyVisible(false);
    }
    setSettingsRouting((current) => {
      if (field === "text_provider") {
        const provider = settingsStatus?.providers?.find((item) => item.id === value);
        return {
          ...current,
          text_provider: value,
          text_model: provider?.default_text_model || "",
          vision_provider: value,
          vision_model: provider?.default_vision_model || "",
          vision_enabled: Boolean(provider?.supports_vision),
        };
      }
      return { ...current, [field]: value };
    });
    setSettingsFeedback(null);
  }

  function changeSettingsCredentialProvider(providerId) {
    const provider = settingsStatus?.providers?.find((item) => item.id === providerId);
    setSettingsCredentialProvider(providerId);
    setSettingsBaseUrl(provider?.base_url || "");
    setSettingsApiKey("");
    setSettingsApiKeyVisible(false);
    setSettingsFeedback(null);
  }

  async function saveApplicationRouting() {
    if (settingsSaving) return;
    const selectedProvider = settingsStatus?.providers?.find(
      (provider) => provider.id === settingsRouting.text_provider,
    );
    if (!selectedProvider?.configured) {
      setSettingsCredentialProvider(settingsRouting.text_provider);
      setSettingsBaseUrl(selectedProvider?.base_url || "");
      setSettingsFeedback({
        scope: "routing",
        tone: "error",
        message: `请先在下方配置并验证 ${selectedProvider?.label || "当前厂商"} API Key。`,
      });
      return;
    }
    const synchronizedRouting = {
      ...settingsRouting,
      vision_enabled: Boolean(settingsRouting.vision_enabled && selectedProvider.supports_vision),
      vision_provider: settingsRouting.text_provider,
      vision_model: selectedProvider.default_vision_model || "",
    };
    setSettingsSaving(true);
    setSettingsFeedback({ scope: "routing", tone: "neutral", message: "正在应用模型配置..." });
    try {
      const response = await fetch("/api/settings/routing", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(synchronizedRouting),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `模型配置保存失败（HTTP ${response.status}）`);
      setSettingsStatus(payload.settings);
      setSettingsRouting(synchronizedRouting);
      setSettingsFeedback({ scope: "routing", tone: "success", message: "模型路由已保存并立即生效。" });
    } catch (error) {
      setSettingsFeedback({
        scope: "routing",
        tone: "error",
        message: error instanceof Error ? error.message : "模型配置保存失败。",
      });
    } finally {
      setSettingsSaving(false);
    }
  }

  async function saveApplicationApiKey(event) {
    event.preventDefault();
    const apiKey = settingsApiKey.trim();
    if (apiKey.length < 10 || settingsSaving) return;
    setSettingsSaving(true);
    setSettingsFeedback({ scope: "credential", tone: "neutral", message: "正在验证厂商凭据..." });
    try {
      const response = await fetch(`/api/settings/providers/${encodeURIComponent(settingsCredentialProvider)}/api-key`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey, base_url: settingsBaseUrl.trim() }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `API Key 保存失败（HTTP ${response.status}）`);
      setSettingsStatus(payload.settings);
      const updatedProvider = payload.settings?.providers?.find((provider) => provider.id === settingsCredentialProvider);
      setSettingsBaseUrl(updatedProvider?.base_url || settingsBaseUrl);
      setSettingsApiKey("");
      setSettingsApiKeyVisible(false);
      const discoveredCount = payload.settings?.validation?.available_model_count || 0;
      setSettingsFeedback({
        scope: "credential",
        tone: "success",
        message: discoveredCount
          ? `API Key 已验证并保存，同时识别到 ${discoveredCount} 个可用模型。`
          : "API Key 已通过真实请求验证并保存。",
      });
      void loadModelHealth(true);
    } catch (error) {
      setSettingsFeedback({
        scope: "credential",
        tone: "error",
        message: error instanceof Error ? error.message : "API Key 验证失败。",
      });
    } finally {
      setSettingsSaving(false);
    }
  }

  async function loadPaperHistory() {
    setHistoryLoading(true);
    try {
      const response = await fetch("/api/history?limit=100");
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `History request failed（HTTP ${response.status}）`);
      }
      const payload = await response.json();
      setHistoryItems(Array.isArray(payload.items) ? payload.items : []);
      setHistoryError("");
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "无法读取论文历史。" );
    } finally {
      setHistoryLoading(false);
    }
  }

  async function loadChatConversation(conversationId, { manageLoading = true } = {}) {
    if (!conversationId) {
      setActiveConversationId("");
      setChatMessages([]);
      return null;
    }
    if (manageLoading) setChatConversationLoading(true);
    try {
      const response = await fetch(`/api/chat/conversations/${encodeURIComponent(conversationId)}`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Conversation restore failed（HTTP ${response.status}）`);
      }
      const payload = await response.json();
      setActiveConversationId(payload.conversation?.id || conversationId);
      setChatMessages(Array.isArray(payload.messages) ? payload.messages : []);
      return payload;
    } finally {
      if (manageLoading) setChatConversationLoading(false);
    }
  }

  async function loadChatConversations(historyId, preferredConversationId = "") {
    if (!historyId) {
      setChatConversations([]);
      setActiveConversationId("");
      setChatMessages([]);
      return [];
    }
    setChatConversationLoading(true);
    try {
      const response = await fetch(`/api/history/${encodeURIComponent(historyId)}/conversations`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Conversation list failed（HTTP ${response.status}）`);
      }
      const payload = await response.json();
      const items = Array.isArray(payload.items) ? payload.items : [];
      setChatConversations(items);
      const targetId = preferredConversationId && items.some((item) => item.id === preferredConversationId)
        ? preferredConversationId
        : items[0]?.id || "";
      if (targetId) {
        await loadChatConversation(targetId, { manageLoading: false });
      } else {
        setActiveConversationId("");
        setChatMessages([]);
      }
      return items;
    } finally {
      setChatConversationLoading(false);
    }
  }

  async function refreshChatConversationList(historyId) {
    if (!historyId) return;
    try {
      const response = await fetch(`/api/history/${encodeURIComponent(historyId)}/conversations`);
      if (!response.ok) return;
      const payload = await response.json();
      if (Array.isArray(payload.items)) setChatConversations(payload.items);
    } catch {
      // The provisional local title remains usable if background refinement fails.
    }
  }

  function resetChatConversations() {
    setChatConversations([]);
    setActiveConversationId("");
    setChatMessages([]);
    setChatConversationLoading(false);
  }

  function startNewChatConversation() {
    chatAbortRef.current?.abort();
    chatAbortRef.current = null;
    setChatStreaming(false);
    setActiveConversationId("");
    setChatMessages([]);
    setChatInput("");
  }

  async function selectChatConversation(conversationId) {
    if (!conversationId) {
      startNewChatConversation();
      return;
    }
    try {
      await loadChatConversation(conversationId);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法恢复这段对话。" );
    }
  }

  async function renameActiveChatConversation(title) {
    if (!activeConversationId || chatStreaming || chatConversationLoading) return false;
    setChatConversationLoading(true);
    try {
      const response = await fetch(`/api/chat/conversations/${encodeURIComponent(activeConversationId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Conversation rename failed（HTTP ${response.status}）`);
      }
      const payload = await response.json();
      if (payload.conversation) {
        setChatConversations((previous) => previous.map((conversation) => (
          conversation.id === payload.conversation.id ? payload.conversation : conversation
        )));
      }
      showToast("会话名称已更新");
      return true;
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法修改会话名称。" );
      return false;
    } finally {
      setChatConversationLoading(false);
    }
  }

  async function deleteActiveChatConversation() {
    if (!activeConversationId || chatStreaming || chatConversationLoading) return;
    const deletingId = activeConversationId;
    setChatConversationLoading(true);
    try {
      const response = await fetch(`/api/chat/conversations/${encodeURIComponent(deletingId)}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Conversation delete failed（HTTP ${response.status}）`);
      }
      const remaining = chatConversations.filter((item) => item.id !== deletingId);
      setChatConversations(remaining);
      if (remaining[0]?.id) {
        await loadChatConversation(remaining[0].id, { manageLoading: false });
      } else {
        setActiveConversationId("");
        setChatMessages([]);
      }
      showToast("对话已删除");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法删除对话。" );
    } finally {
      setChatConversationLoading(false);
    }
  }

  async function openHistoryPaper(item) {
    if (!item?.id || historyBusyId) return;
    setHistoryBusyId(item.id);
    try {
      const response = await fetch(`/api/history/${encodeURIComponent(item.id)}`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `History restore failed（HTTP ${response.status}）`);
      }
      const payload = await response.json();
      chatAbortRef.current?.abort();
      chatAbortRef.current = null;
      setSelectedFile(null);
      setSelectedChapterIndex(0);
      setActiveTab("概览");
      setAnalysisData(payload);
      setAnalysisError("");
      setIsAnalyzing(false);
      setAgentStates(completeAgentStates);
      setAgentStreams(emptyAgentStreams);
      setStreamMessage("已从本地历史恢复完整论文分析。" );
      setSelectionAction(null);
      setChatOpen(false);
      setChatQuote("");
      setChatInput("");
      resetChatConversations();
      setChatStreaming(false);
      setHistoryOpen(false);
      setRecentOpen(true);
      setWorkspaceMode("reading");
      showToast(`已打开 ${item.title}`);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法打开历史论文。" );
    } finally {
      setHistoryBusyId("");
    }
  }

  async function deleteHistoryPaper(item) {
    if (!item?.id || historyBusyId) return;
    setHistoryBusyId(item.id);
    try {
      const response = await fetch(`/api/history/${encodeURIComponent(item.id)}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `History delete failed（HTTP ${response.status}）`);
      }
      if (displayedData.history_id === item.id) {
        setAnalysisData(sampleAnalysis);
        setSelectedFile(null);
        setAgentStates(emptyAgentStates);
        setAgentStreams(emptyAgentStreams);
        setStreamMessage("已准备好开始分析");
        setChatOpen(false);
        resetChatConversations();
      }
      await loadPaperHistory();
      showToast("历史论文已删除");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法删除历史论文。" );
    } finally {
      setHistoryBusyId("");
    }
  }

  function chooseFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setAnalysisError("Please select a PDF file.");
      showToast("PDF files only");
      return;
    }
    setSelectedFile(file);
    setSelectedChapterIndex(0);
    setAnalysisError("");
    setAgentStates(emptyAgentStates);
    setAgentStreams(emptyAgentStreams);
    setStreamMessage("PDF 已选择，可以开始流式分析。");
    setAnalysisData(pendingAnalysisForFile(file));
    chatAbortRef.current?.abort();
    chatAbortRef.current = null;
    setChatStreaming(false);
    setSelectionAction(null);
    setChatOpen(false);
    setChatQuote("");
    setChatInput("");
    resetChatConversations();
    showToast("PDF ready for analysis");
  }

  function handleResultSelection() {
    window.requestAnimationFrame(() => {
      const selection = window.getSelection();
      const panel = resultsPanelRef.current;
      const scroll = resultsScrollRef.current;
      if (!selection || selection.isCollapsed || !selection.rangeCount || !panel || !scroll) {
        setSelectionAction(null);
        return;
      }

      const range = selection.getRangeAt(0);
      const ancestor = range.commonAncestorContainer.nodeType === Node.TEXT_NODE
        ? range.commonAncestorContainer.parentElement
        : range.commonAncestorContainer;
      if (!ancestor || !scroll.contains(ancestor)) {
        setSelectionAction(null);
        return;
      }

      const text = selection.toString().replace(/\s+/g, " ").trim().slice(0, 4000);
      if (text.length < 2) {
        setSelectionAction(null);
        return;
      }

      const rect = range.getBoundingClientRect();
      const panelRect = panel.getBoundingClientRect();
      const left = Math.min(Math.max(rect.left + rect.width / 2 - panelRect.left, 108), panelRect.width - 108);
      let top = rect.top - panelRect.top - 44;
      if (top < 58) top = rect.bottom - panelRect.top + 8;
      setSelectionAction({ text, left, top });
    });
  }

  async function openPaperChat(quote = "") {
    setChatQuote(quote);
    setChatOpen(true);
    setSelectionAction(null);
    window.getSelection()?.removeAllRanges();
    if (!displayedData.history_id) {
      resetChatConversations();
      return;
    }
    try {
      await loadChatConversations(displayedData.history_id, activeConversationId);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法读取论文追问记录。" );
    }
  }

  function openChatFromSelection() {
    if (!selectionAction?.text) return;
    void openPaperChat(selectionAction.text);
  }

  function openChatDirectly() {
    void openPaperChat("");
  }

  async function sendChatMessage() {
    const question = chatInput.trim();
    if (!question || chatStreaming) return;

    const quote = chatQuote.trim();
    const userId = `user-${Date.now()}`;
    const assistantId = `assistant-${Date.now()}`;
    setChatMessages((previous) => [
      ...previous,
      { id: userId, role: "user", content: question, quote },
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setChatInput("");
    setChatQuote("");
    setChatStreaming(true);
    const controller = new AbortController();
    chatAbortRef.current?.abort();
    chatAbortRef.current = controller;

    try {
      const useDemoChat = displayedData.mode !== "live";
      const response = await fetch(`/api/chat/stream?demo=${useDemoChat ? "true" : "false"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          question,
          analysis_id: displayedData.analysis_id || null,
          history_id: displayedData.history_id || null,
          conversation_id: activeConversationId || null,
          selected_text: quote || null,
          context: analysisContextForChat(displayedData),
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `追问请求失败（HTTP ${response.status}）`);
      }
      if (!response.body) throw new Error("当前浏览器无法读取流式回答。");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      let completed = false;
      let completionEvent = null;

      function handleChatEvent(event) {
        if (event.type === "error") {
          if (event.conversation_id) setActiveConversationId(event.conversation_id);
          throw new Error(event.message || "追问失败。");
        }
        if (event.type === "token") {
          answer += event.text || "";
          setChatMessages((previous) => previous.map((message) => (
            message.id === assistantId ? { ...message, content: answer } : message
          )));
        }
        if (event.type === "complete") {
          completed = true;
          completionEvent = event;
        }
      }

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          handleChatEvent(JSON.parse(line));
        }
        if (done) break;
      }

      if (buffer.trim()) {
        handleChatEvent(JSON.parse(buffer));
      }
      if (!completed || !answer.trim()) throw new Error("回答在完成前意外结束。");
      setChatMessages((previous) => previous.map((message) => (
        message.id === userId && completionEvent?.user_message
          ? completionEvent.user_message
          : message.id === assistantId
            ? completionEvent?.assistant_message || {
              ...message,
              content: answer,
              model_trace: completionEvent?.model_trace || null,
            }
            : message
      )));
      if (completionEvent?.conversation_id) {
        setActiveConversationId(completionEvent.conversation_id);
      }
      if (completionEvent?.conversation) {
        setChatConversations((previous) => [
          completionEvent.conversation,
          ...previous.filter((item) => item.id !== completionEvent.conversation.id),
        ]);
      }
      if (completionEvent?.title_generation_scheduled && displayedData.history_id) {
        conversationTitleRefreshDelays.forEach((delay) => {
          window.setTimeout(() => void refreshChatConversationList(displayedData.history_id), delay);
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "追问失败。";
      setChatMessages((previous) => previous.map((item) => (
        item.id === assistantId ? { ...item, content: message, error: true } : item
      )));
    } finally {
      if (chatAbortRef.current === controller) {
        chatAbortRef.current = null;
        setChatStreaming(false);
      }
    }
  }

  function applyStreamEvent(event) {
    const agentNames = {
      method: "方法分析 Agent",
      experiment: "实验分析 Agent",
      critic: "批判性评审 Agent",
      summary: "总结 Agent",
    };

    if (event.type === "paper") {
      setSelectedChapterIndex(0);
      setAnalysisData((previous) => ({
        ...(previous || {}),
        mode: event.mode || previous?.mode || "live",
        paper: event.paper || previous?.paper,
      }));
      setStreamMessage(`已解析 ${event.paper?.sections_count ?? "若干"} 个章节，正在启动 Agent。`);
      return "";
    }

    if (event.type === "section_titles_started") {
      setStreamMessage(`正在将 ${event.count ?? "若干"} 个英文章节标题统一翻译为中文。`);
      return "";
    }

    if (event.type === "section_titles_complete") {
      setStreamMessage(`已翻译 ${event.translated ?? 0} 个自定义章节标题，正在整理论文结构。`);
      return "";
    }

    if (event.type === "section_titles_error") {
      setStreamMessage("部分自定义章节标题翻译失败，已继续使用本地中文词典处理。" );
      return "";
    }

    if (event.type === "vision_started") {
      setStreamMessage("正在渲染 PDF 图表并调用视觉模型生成 F 类证据。");
      return "";
    }

    if (event.type === "vision_complete") {
      setStreamMessage(`已生成 ${event.enriched ?? 0} 个视觉摘要，正在建立文本/表格/图像证据索引。`);
      return "";
    }

    if (event.type === "vision_error") {
      setStreamMessage("视觉摘要生成失败，已自动继续使用正文、表格和图注证据。");
      return "";
    }

    if (event.type === "evidence_index") {
      setAnalysisData((previous) => ({
        ...(previous || {}),
        evidence_index: event.evidence_index || [],
      }));
      setStreamMessage(`已建立 ${(event.evidence_index || []).length} 个文本/表格/图像证据片段，正在进行证据化研读。`);
      return "";
    }

    if (event.type === "agent_started") {
      setAgentStates((previous) => ({ ...previous, [event.agent]: "running" }));
      setAgentStreams((previous) => ({ ...previous, [event.agent]: "" }));
      setStreamMessage(`${agentNames[event.agent] || event.agent} 正在阅读相关章节。`);
      return "";
    }

    if (event.type === "agent_token") {
      setAgentStreams((previous) => {
        const current = `${previous[event.agent] || ""}${event.text || ""}`;
        return { ...previous, [event.agent]: current.slice(-2600) };
      });
      return "";
    }

    if (event.type === "agent_complete") {
      setAgentStates((previous) => ({ ...previous, [event.agent]: "complete" }));
      setAnalysisData((previous) => ({
        ...(previous || {}),
        [event.output_key]: event.output,
      }));
      setStreamMessage(`${agentNames[event.agent] || event.agent} 已完成，正在展示阶段性结果。`);
      return "";
    }

    if (event.type === "complete") {
      const { type, ...payload } = event;
      setAnalysisData(payload);
      setAgentStates(completeAgentStates);
      setAgentStreams(emptyAgentStreams);
      setStreamMessage("所有 Agent 已完成，最终研读笔记已生成。" );
      if (payload.history_id) void loadPaperHistory();
      return "";
    }

    if (event.type === "history_error") {
      showToast("分析已完成，但本地历史保存失败。" );
      return "";
    }

    if (event.type === "error") {
      setAgentStates((previous) => ({
        ...previous,
        ...(event.agent ? { [event.agent]: "failed" } : {}),
      }));
      return event.message || "Analysis failed.";
    }

    return "";
  }

  async function startAnalysis() {
    if (!selectedFile) {
      setAnalysisError("Select or drop a PDF first, then run analysis.");
      showToast("Select a PDF first");
      return;
    }

    setActiveTab("概览");
    setIsAnalyzing(true);
    setHistoryOpen(false);
    setAnalysisError("");
    setAgentStates(emptyAgentStates);
    setAgentStreams(emptyAgentStreams);
    setStreamMessage("正在将 PDF 上传到后端..." );
    setAnalysisData(pendingAnalysisForFile(selectedFile));
    chatAbortRef.current?.abort();
    chatAbortRef.current = null;
    setChatStreaming(false);
    setSelectionAction(null);
    setChatOpen(false);
    setChatQuote("");
    setChatInput("");
    resetChatConversations();
    showToast("Live analysis started");

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const response = await fetch("/api/analyze/stream", {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Analysis failed with HTTP ${response.status}`);
      }
      if (!response.body) {
        throw new Error("This browser does not expose a readable response stream.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          const streamError = applyStreamEvent(event);
          if (streamError) {
            throw new Error(streamError);
          }
          if (event.type === "complete") {
            completed = true;
          }
        }

        if (done) break;
      }

      if (buffer.trim()) {
        const event = JSON.parse(buffer);
        const streamError = applyStreamEvent(event);
        if (streamError) {
          throw new Error(streamError);
        }
        if (event.type === "complete") {
          completed = true;
        }
      }

      if (!completed) {
        throw new Error("流式分析在生成最终结果前意外结束。" );
      }
      showToast("Live analysis complete");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Analysis failed.";
      setAnalysisError(message);
      setStreamMessage(message);
      showToast("Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  }

  function copyJson() {
    const exportData = workspaceMode === "comparison" ? comparisonData : displayedData;
    if (!exportData) return;
    const payload = JSON.stringify(exportData, null, 2);
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(payload).catch(() => {});
    }
    showToast("Copied JSON to clipboard");
  }

  function downloadNotes(format) {
    const exportData = workspaceMode === "comparison" ? comparisonData : displayedData;
    if (!exportData) return;
    const ext = format === "markdown" ? "md" : "json";
    const text = format === "markdown"
      ? workspaceMode === "comparison"
        ? comparisonMarkdownFromData(exportData)
        : markdownFromAnalysis(displayedData)
      : JSON.stringify(exportData, null, 2);
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${workspaceMode === "comparison" ? "paper-comparison" : "paper-reader-notes"}.${ext}`;
    link.click();
    URL.revokeObjectURL(url);
    showToast(format === "markdown" ? "Markdown exported" : "Notes downloaded");
  }

  return (
    <div className="app-shell">
      <div className="background-wash" />

      <header className="topbar glass">
        <div className="brand">
          <span className="brand-mark"><IconBook2 size={18} stroke={1.8} /></span>
          <span>Paper Reader</span>
        </div>
        <div className={`workspace-menu ${workspaceMenuOpen ? "open" : ""}`} ref={workspaceMenuRef}>
          <button
            className="workspace-switch"
            type="button"
            aria-haspopup="menu"
            aria-expanded={workspaceMenuOpen}
            onClick={() => setWorkspaceMenuOpen((value) => !value)}
          >
            {workspaceMode === "comparison" ? "Comparison Workspace" : "Reading Workspace"}
            <IconChevronDown size={16} stroke={1.8} />
          </button>
          <div className="workspace-dropdown glass" role="menu">
            <button
              className={workspaceMode === "reading" ? "active" : ""}
              type="button"
              role="menuitem"
              onClick={() => {
                setWorkspaceMode("reading");
                setWorkspaceMenuOpen(false);
                setHistoryOpen(false);
              }}
            >
              <IconBook2 size={17} />
              <span><strong>单篇论文研读</strong><small>Reading Workspace</small></span>
              {workspaceMode === "reading" && <IconCheck size={16} />}
            </button>
            <button
              className={workspaceMode === "comparison" ? "active" : ""}
              type="button"
              role="menuitem"
              onClick={() => {
                setWorkspaceMode("comparison");
                setWorkspaceMenuOpen(false);
                setHistoryOpen(false);
                if (!historyItems.length) void loadPaperHistory();
              }}
            >
              <IconShare3 size={17} />
              <span><strong>多论文对比</strong><small>Comparison Workspace</small></span>
              {workspaceMode === "comparison" && <IconCheck size={16} />}
            </button>
          </div>
        </div>
        <nav className="top-actions">
          <button
            type="button"
            onClick={() => {
              if (!historyOpen) void loadPaperHistory();
              setHistoryOpen((value) => !value);
            }}
          >
            <IconHistory size={18} stroke={1.8} /> History
          </button>
          <div className="export-menu">
            <button
              className="export-trigger"
              type="button"
              aria-haspopup="menu"
              aria-label="Export options"
            >
              <IconShare3 size={18} stroke={1.8} /> Export
              <IconChevronDown size={14} stroke={1.8} />
            </button>
            <div className="export-dropdown glass" role="menu">
              <button type="button" role="menuitem" onClick={copyJson} disabled={!hasExportData}>
                <IconCopy size={17} /> 复制 JSON
              </button>
              <button
                type="button"
                role="menuitem"
                onClick={() => downloadNotes("markdown")}
                disabled={!hasExportData}
              >
                <IconMarkdown size={17} /> 导出 Markdown
              </button>
              <button
                type="button"
                role="menuitem"
                onClick={() => downloadNotes("json")}
                disabled={!hasExportData}
              >
                <IconDownload size={17} /> 下载笔记
              </button>
            </div>
          </div>
          <button type="button" onClick={openApplicationSettings}>
            <IconSettings size={18} stroke={1.8} /> Settings
          </button>
          <img src={avatarUrl} alt="Researcher profile" className="avatar" />
          <IconChevronDown size={16} stroke={1.7} />
        </nav>
      </header>

      {workspaceMode === "comparison" ? (
        <ComparisonWorkspace
          historyItems={historyItems}
          historyLoading={historyLoading}
          historyError={historyError}
          showToast={showToast}
          onResultChange={setComparisonData}
          modelLabel={activeModelLabel}
          onAddPaper={() => {
            setWorkspaceMode("reading");
            window.setTimeout(() => fileInputRef.current?.click(), 80);
          }}
        />
      ) : (
      <main className="workspace">
        <aside className="paper-panel glass">
          <label
            className={`dropzone ${dragActive ? "dragging" : ""}`}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragActive(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragActive(false);
              chooseFile(event.dataTransfer.files?.[0]);
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />
            <IconFileDescription size={34} stroke={1.4} />
            <span>Drop a research paper to begin</span>
            <small>PDF up to 200 MB</small>
            <button type="button" onClick={() => fileInputRef.current?.click()}>
              Select PDF
            </button>
          </label>

          <section className="uploaded-card">
            <div className="section-heading">
              <span>Uploaded Paper</span>
              <small><i /> {displayedData.history_id ? "Saved" : selectedFile ? "Ready" : "Sample"}</small>
            </div>
            <div className="paper-card">
              <IconFileTypePdf className="pdf-icon" size={34} stroke={1.6} />
              <div>
                <h2>{displayedPaper.title}</h2>
                <p>{displayedPaper.filename || "Sample PDF"}</p>
              </div>
              <div className="paper-stats">
                <span><IconFileAnalytics size={15} /> {displayedPaper.pages || "—"}<small>Pages</small></span>
                <span><IconListDetails size={15} /> {displayedPaper.sections_count || "—"}<small>Sections</small></span>
                <span><IconCloudUpload size={15} /> {formatBytes(displayedPaper.size_bytes)}<small>File Size</small></span>
              </div>
            </div>
          </section>

          <section className={`chapters ${chaptersOpen ? "expanded" : "collapsed"}`}>
            <button
              className="chapters-toggle"
              type="button"
              aria-expanded={chaptersOpen}
              aria-controls="chapter-list"
              onClick={() => setChaptersOpen((value) => !value)}
            >
              <span>章节目录</span>
              <IconChevronDown className="chapters-chevron" size={16} />
            </button>
            <div className="chapters-content" id="chapter-list">
              {displaySections.map((chapter, index) => (
                <button
                  className={`${selectedChapterIndex === index ? "active" : ""} ${chapter.status?.tone || "ready"}`}
                  key={`${chapter.title}-${index}`}
                  type="button"
                  onClick={() => setSelectedChapterIndex(index)}
                >
                  <span className="chapter-index">{index + 1}</span>
                  <span className="chapter-copy">
                    <strong>{chapter.displayTitle || cleanChapterTitle(chapter, index)}</strong>
                    <small>{chapter.meta || chapterMeta(chapter)}</small>
                  </span>
                  <span className={`chapter-status ${chapter.status?.tone || "ready"}`}>
                    {chapter.status?.icon === "check" ? (
                      <IconCheck size={12} stroke={2.5} />
                    ) : (
                      <i />
                    )}
                    {chapter.status?.label || "已识别"}
                  </span>
                </button>
              ))}
            </div>
          </section>

          <section className={`recent ${recentOpen ? "expanded" : "collapsed"}`}>
            <button
              className="recent-toggle"
              type="button"
              aria-expanded={recentOpen}
              aria-controls="recent-papers-list"
              onClick={() => {
                if (!recentOpen) void loadPaperHistory();
                setRecentOpen((value) => !value);
              }}
            >
              <span>Recent Papers</span>
              <IconChevronDown className="recent-chevron" size={16} />
            </button>
            <div className="recent-content" id="recent-papers-list">
              {historyLoading && <div className="history-empty">正在读取历史...</div>}
              {!historyLoading && historyError && <div className="history-empty error">{historyError}</div>}
              {!historyLoading && !historyError && !historyItems.length && (
                <div className="history-empty">暂无已保存论文</div>
              )}
              {!historyLoading && historyItems.map((item) => (
                <HistoryPaperButton
                  key={item.id}
                  item={item}
                  active={displayedData.history_id === item.id}
                  disabled={historyBusyId === item.id}
                  onOpen={openHistoryPaper}
                />
              ))}
              {historyItems.length > 0 && (
                <button className="history-link" type="button" onClick={() => setHistoryOpen(true)}>
                  管理全部历史 <IconChevronRight size={16} />
                </button>
              )}
            </div>
          </section>
        </aside>

        <section className="agent-stage">
          <div className="agent-map" aria-label="Agent analysis workflow">
            <svg className="connectors" viewBox="0 0 640 620" aria-hidden="true">
              <path className="flow-line top-line" d="M320 150 L320 308" />
              <path className="flow-line left-line" d="M205 360 C205 432 320 420 320 500" />
              <path className="flow-line right-line" d="M435 360 C435 432 320 420 320 500" />
              <circle className="flow-node" cx="320" cy="308" r="4" />
              <circle className="flow-node" cx="320" cy="500" r="4" />
            </svg>
            {agents.map((agent) => (
              <AgentCard key={agent.id} agent={agent} />
            ))}
          </div>
          <div className="run-pill glass">
            <span className={workflowDotClass} />
            {workflowLabel}
            <IconClock size={16} stroke={1.8} />
            <span>{workflowDetail}</span>
          </div>
          <AppButton className="analyze-button" onClick={startAnalysis} disabled={isAnalyzing}>
            {isAnalyzing ? <IconLoader2 className="spin" size={20} /> : <IconSparkles size={20} />}
            {isAnalyzing ? "Analyzing Paper" : "Analyze Paper"}
          </AppButton>
        </section>

        <section className={`results-panel glass${chatOpen ? " chat-open" : ""}`} ref={resultsPanelRef}>
          <div className="tabs">
            {tabs.map((tab) => (
              <button
                className={activeTab === tab ? "active" : ""}
                key={tab}
                type="button"
                onClick={() => { setActiveTab(tab); setSelectionAction(null); }}
              >
                {tab}
              </button>
            ))}
          </div>
          <div
            className="results-scroll"
            ref={resultsScrollRef}
            onMouseUp={handleResultSelection}
            onKeyUp={handleResultSelection}
            onScroll={() => setSelectionAction(null)}
          >
            <ResultContent
              activeTab={activeTab}
              data={displayedData}
              error={analysisError}
              streamMessage={streamMessage}
              agentStreams={agentStreams}
              isAnalyzing={isAnalyzing}
            />
          </div>
          {selectionAction && !chatOpen && (
            <button
              className="selection-chat-action"
              type="button"
              style={{ left: selectionAction.left, top: selectionAction.top }}
              onMouseDown={(event) => event.preventDefault()}
              onClick={openChatFromSelection}
            >
              <IconMessageCircle size={16} stroke={1.8} /> 在侧边聊天中提问
            </button>
          )}
          {!chatOpen && (
            <div className="results-chat-launcher">
              <button
                className="open-paper-chat"
                type="button"
                aria-label="打开论文追问"
                aria-controls="paper-chat-drawer"
                data-tooltip="打开论文追问"
                onClick={openChatDirectly}
              >
                <IconSparkles size={19} stroke={1.9} />
              </button>
            </div>
          )}
          {chatOpen && (
            <PaperChatDrawer
              paperTitle={displayedPaper.title || "当前论文"}
              modelLabel={activeModelLabel}
              conversations={chatConversations}
              activeConversationId={activeConversationId}
              messages={chatMessages}
              input={chatInput}
              quote={chatQuote}
              isStreaming={chatStreaming}
              isConversationLoading={chatConversationLoading}
              onInputChange={setChatInput}
              onClearQuote={() => setChatQuote("")}
              onSend={sendChatMessage}
              onClose={() => setChatOpen(false)}
              onConversationChange={selectChatConversation}
              onNewConversation={startNewChatConversation}
              onDeleteConversation={deleteActiveChatConversation}
              onRenameConversation={renameActiveChatConversation}
            />
          )}
        </section>
      </main>
      )}

      {settingsOpen && (
        <SettingsDialog
          status={settingsStatus}
          loading={settingsLoading}
          error={settingsError}
          modelHealth={settingsModelHealth}
          healthLoading={settingsHealthLoading}
          healthError={settingsHealthError}
          routing={settingsRouting}
          credentialProvider={settingsCredentialProvider}
          baseUrl={settingsBaseUrl}
          apiKey={settingsApiKey}
          apiKeyVisible={settingsApiKeyVisible}
          saving={settingsSaving}
          feedback={settingsFeedback}
          onRoutingChange={changeSettingsRouting}
          onSaveRouting={() => void saveApplicationRouting()}
          onCredentialProviderChange={changeSettingsCredentialProvider}
          onBaseUrlChange={(value) => {
            setSettingsBaseUrl(value);
            if (settingsFeedback) setSettingsFeedback(null);
          }}
          onApiKeyChange={(value) => {
            setSettingsApiKey(value);
            if (settingsFeedback) setSettingsFeedback(null);
          }}
          onToggleApiKey={() => setSettingsApiKeyVisible((value) => !value)}
          onSaveApiKey={saveApplicationApiKey}
          onRefreshHealth={() => void loadModelHealth(true)}
          onRetry={() => void loadApplicationSettings()}
          onClose={closeApplicationSettings}
        />
      )}

      {historyOpen && (
        <div className="history-popover glass">
          <div className="popover-heading">
            <strong>论文历史</strong>
            <button type="button" onClick={() => setHistoryOpen(false)}><IconX size={16} /></button>
          </div>
          <div className="history-popover-list">
            {historyLoading && <div className="history-empty">正在读取历史...</div>}
            {!historyLoading && historyError && <div className="history-empty error">{historyError}</div>}
            {!historyLoading && !historyError && !historyItems.length && (
              <div className="history-empty">暂无已保存论文</div>
            )}
            {historyItems.map((item) => (
              <div className="history-popover-row" key={item.id}>
                <HistoryPaperButton
                  item={item}
                  active={displayedData.history_id === item.id}
                  disabled={historyBusyId === item.id}
                  onOpen={openHistoryPaper}
                />
                <button
                  className="history-delete"
                  type="button"
                  title="删除历史论文"
                  aria-label={`删除 ${item.title}`}
                  disabled={historyBusyId === item.id}
                  onClick={() => deleteHistoryPaper(item)}
                >
                  <IconTrash size={15} stroke={1.8} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {toast && <div className="toast glass"><IconCheck size={18} /> {toast}</div>}
    </div>
  );
}

function AgentCard({ agent }) {
  const Icon = agent.icon;
  return (
    <article
      className={`agent-card glass ${agent.state} ${agent.complete ? "complete" : ""} ${agent.active ? "active" : ""} ${agent.failed ? "failed" : ""}`}
      data-agent-id={agent.id}
      style={{ left: `${agent.x}%`, top: `${agent.y}px` }}
    >
      <div className={`agent-icon ${agent.accent}`}>
        <Icon size={29} stroke={1.7} />
      </div>
      <div className="agent-copy">
        <header>
          <h2>{agent.name}</h2>
          <span className={`agent-status ${agent.statusTone}`}><i /> {agent.status}</span>
        </header>
        <p>{agent.summary}</p>
      </div>
      <div className="steps">
        {agent.steps.map((step) => (
          <span className={`step ${step.state}`} key={step.label}>
            {step.state === "done" && <IconCheck size={14} />}
            {step.state === "active" && <i className="step-current" />}
            {step.state === "failed" && <IconAlertCircle size={14} />}
            {step.state === "pending" && <i className="step-empty" />}
            <small>{step.label}</small>
          </span>
        ))}
      </div>
      <div className="progress-track">
        <div style={{ width: `${agent.progress}%` }} />
      </div>
    </article>
  );
}
