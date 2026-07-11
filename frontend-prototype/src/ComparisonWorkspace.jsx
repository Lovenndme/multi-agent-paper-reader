import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  IconAlertCircle,
  IconArrowsLeftRight,
  IconCheck,
  IconChevronRight,
  IconFileTypePdf,
  IconHistory,
  IconLoader2,
  IconMessageCircle,
  IconPencil,
  IconPlus,
  IconQuote,
  IconSend,
  IconSparkles,
  IconTrash,
  IconX,
} from "@tabler/icons-react";

const focusOptions = [
  { value: "comprehensive", label: "综合" },
  { value: "method", label: "方法" },
  { value: "experiment", label: "实验" },
  { value: "critique", label: "评审" },
  { value: "custom", label: "自定义" },
];

const resultTabs = [
  { value: "overview", label: "对比概览" },
  { value: "method", label: "方法与架构" },
  { value: "experiment", label: "实验结果" },
  { value: "critique", label: "创新与局限" },
  { value: "gaps", label: "研究空白" },
];

const comparabilityLabels = {
  direct: "可直接比较",
  conditional: "需结合条件",
  not_comparable: "不宜直接比较",
};

const chatMarkdownComponents = {
  a: ({ children, href }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
  table: ({ children }) => <div className="chat-table-scroll"><table>{children}</table></div>,
};

function formatAge(value) {
  const timestamp = Date.parse(value || "");
  if (!Number.isFinite(timestamp)) return "已保存";
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 60) return "刚刚";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)} 天前`;
  return new Date(timestamp).toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
}

function escapeMarkdownCell(value) {
  return String(value || "").replace(/\|/g, "\\|").replace(/\n+/g, " ");
}

export function comparisonMarkdownFromData(data) {
  const comparison = data?.comparison;
  if (!comparison) return "";
  const lines = [
    `# ${comparison.title}`,
    "",
    `> 对比重点：${comparison.focus}`,
    "",
    "## 论文",
    ...comparison.papers.map((paper) => `- **${paper.label}** ${paper.title}`),
    "",
    "## 综合结论",
    comparison.executive_summary,
    "",
    "## 对比矩阵",
  ];
  const headers = ["维度", ...comparison.papers.map((paper) => `${paper.label} ${paper.title}`), "跨论文结论"];
  lines.push(`| ${headers.map(escapeMarkdownCell).join(" | ")} |`);
  lines.push(`| ${headers.map(() => "---").join(" | ")} |`);
  for (const dimension of comparison.dimensions) {
    const byLabel = Object.fromEntries(dimension.cells.map((cell) => [cell.paper_label, cell]));
    const row = [
      dimension.title,
      ...comparison.papers.map((paper) => {
        const cell = byLabel[paper.label];
        const refs = cell?.evidence_ids?.length ? ` ${cell.evidence_ids.map((id) => `[${id}]`).join(" ")}` : "";
        return `${cell?.summary || "证据不足"}${refs}`;
      }),
      `${dimension.synthesis}${dimension.warning ? `（${dimension.warning}）` : ""}`,
    ];
    lines.push(`| ${row.map(escapeMarkdownCell).join(" | ")} |`);
  }
  const sections = [
    ["共同点", comparison.common_ground],
    ["关键差异", comparison.key_differences],
    ["研究空白", comparison.research_gaps],
    ["适用建议", comparison.recommendations],
    ["注意事项", [...(comparison.warnings || []), ...(data.assessment?.warnings || [])]],
  ];
  for (const [title, items] of sections) {
    if (!items?.length) continue;
    lines.push("", `## ${title}`, ...items.map((item) => `- ${item}`));
  }
  return lines.join("\n");
}

export function ComparisonWorkspace({
  historyItems,
  historyLoading,
  historyError,
  showToast,
  onResultChange,
  onAddPaper,
}) {
  const [selectedIds, setSelectedIds] = useState([]);
  const [focus, setFocus] = useState("comprehensive");
  const [customFocus, setCustomFocus] = useState("");
  const [activeTab, setActiveTab] = useState("overview");
  const [isComparing, setIsComparing] = useState(false);
  const [progress, setProgress] = useState("选择论文后开始对比");
  const [loadedLabels, setLoadedLabels] = useState([]);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  const [comparisonId, setComparisonId] = useState("");
  const [savedItems, setSavedItems] = useState([]);
  const [savedLoading, setSavedLoading] = useState(true);
  const [savedOpen, setSavedOpen] = useState(true);
  const [busyId, setBusyId] = useState("");
  const [selectedEvidence, setSelectedEvidence] = useState(null);
  const [selectionAction, setSelectionAction] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatQuote, setChatQuote] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatStreaming, setChatStreaming] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState("");
  const [conversationLoading, setConversationLoading] = useState(false);
  const resultPanelRef = useRef(null);
  const resultScrollRef = useRef(null);
  const compareAbortRef = useRef(null);
  const chatAbortRef = useRef(null);

  useEffect(() => {
    void loadSavedComparisons();
    return () => {
      compareAbortRef.current?.abort();
      chatAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    onResultChange?.(data);
  }, [data, onResultChange]);

  const comparison = data?.comparison;
  const assessment = data?.assessment;
  const paperLabels = useMemo(
    () => Object.fromEntries(selectedIds.map((id, index) => [id, `P${index + 1}`])),
    [selectedIds],
  );
  const evidenceById = useMemo(
    () => Object.fromEntries((data?.evidence_catalog || []).map((item) => [item.id, item])),
    [data],
  );

  async function loadSavedComparisons() {
    setSavedLoading(true);
    try {
      const response = await fetch("/api/comparisons?limit=100");
      if (!response.ok) throw new Error(`无法读取对比历史（HTTP ${response.status}）`);
      const payload = await response.json();
      setSavedItems(Array.isArray(payload.items) ? payload.items : []);
    } catch (loadError) {
      showToast(loadError instanceof Error ? loadError.message : "无法读取对比历史。" );
    } finally {
      setSavedLoading(false);
    }
  }

  function togglePaper(historyId) {
    setSelectedIds((previous) => {
      if (previous.includes(historyId)) return previous.filter((id) => id !== historyId);
      if (previous.length >= 4) {
        showToast("一次最多比较 4 篇论文");
        return previous;
      }
      return [...previous, historyId];
    });
  }

  async function startComparison() {
    if (selectedIds.length < 2 || isComparing) {
      showToast("请至少选择 2 篇论文");
      return;
    }
    if (focus === "custom" && !customFocus.trim()) {
      showToast("请填写自定义对比问题");
      return;
    }
    setIsComparing(true);
    setError("");
    setData(null);
    setComparisonId("");
    setLoadedLabels([]);
    setProgress("正在读取历史论文与证据");
    setSelectedEvidence(null);
    setChatOpen(false);
    resetChat();
    const controller = new AbortController();
    compareAbortRef.current?.abort();
    compareAbortRef.current = controller;
    try {
      const selectedPapers = historyItems.filter((item) => selectedIds.includes(item.id));
      const useDemo = selectedPapers.length > 0 && selectedPapers.every((item) => item.mode === "demo");
      const response = await fetch(`/api/comparisons/stream?demo=${useDemo ? "true" : "false"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          history_ids: selectedIds,
          focus,
          custom_focus: focus === "custom" ? customFocus.trim() : null,
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `对比请求失败（HTTP ${response.status}）`);
      }
      if (!response.body) throw new Error("当前浏览器无法读取流式对比结果。");
      await readNdjson(response.body, (event) => {
        if (event.type === "error") throw new Error(event.message || "多论文对比失败。");
        if (event.type === "paper_loaded") {
          setLoadedLabels((previous) => [...new Set([...previous, event.label])]);
          setProgress(`${event.label} 已载入，正在对齐证据`);
        }
        if (event.type === "comparison_token") setProgress("GLM-5.2 正在生成对比矩阵");
        if (event.type === "complete") {
          setData(event);
          setComparisonId(event.comparison_id || "");
          setProgress("对比完成");
          setActiveTab("overview");
        }
      });
      await loadSavedComparisons();
      showToast("多论文对比已完成并保存");
    } catch (compareError) {
      if (compareError?.name === "AbortError") return;
      const message = compareError instanceof Error ? compareError.message : "多论文对比失败。";
      setError(message);
      setProgress("对比需要处理");
    } finally {
      if (compareAbortRef.current === controller) compareAbortRef.current = null;
      setIsComparing(false);
    }
  }

  async function openSavedComparison(item) {
    if (!item?.id || busyId) return;
    setBusyId(item.id);
    try {
      const response = await fetch(`/api/comparisons/${encodeURIComponent(item.id)}`);
      if (!response.ok) throw new Error(`无法恢复对比结果（HTTP ${response.status}）`);
      const payload = await response.json();
      setData(payload);
      setComparisonId(payload.comparison_id || item.id);
      setSelectedIds((payload.comparison?.papers || []).map((paper) => paper.history_id));
      setFocus(payload.workspace?.focus || "comprehensive");
      setCustomFocus(payload.workspace?.custom_focus || "");
      setActiveTab("overview");
      setError("");
      setProgress("已从本地恢复对比结果");
      setSelectedEvidence(null);
      setChatOpen(false);
      resetChat();
      showToast(`已打开 ${item.title}`);
    } catch (openError) {
      showToast(openError instanceof Error ? openError.message : "无法恢复对比结果。" );
    } finally {
      setBusyId("");
    }
  }

  async function deleteSavedComparison(item) {
    if (!item?.id || busyId) return;
    setBusyId(item.id);
    try {
      const response = await fetch(`/api/comparisons/${encodeURIComponent(item.id)}`, { method: "DELETE" });
      if (!response.ok) throw new Error(`无法删除对比结果（HTTP ${response.status}）`);
      if (comparisonId === item.id) {
        setData(null);
        setComparisonId("");
        setChatOpen(false);
        resetChat();
      }
      await loadSavedComparisons();
      showToast("对比记录已删除");
    } catch (deleteError) {
      showToast(deleteError instanceof Error ? deleteError.message : "无法删除对比记录。" );
    } finally {
      setBusyId("");
    }
  }

  function handleResultSelection() {
    window.requestAnimationFrame(() => {
      const selection = window.getSelection();
      const panel = resultPanelRef.current;
      const scroll = resultScrollRef.current;
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
      if (text.length < 2) return setSelectionAction(null);
      const rect = range.getBoundingClientRect();
      const panelRect = panel.getBoundingClientRect();
      const left = Math.min(Math.max(rect.left + rect.width / 2 - panelRect.left, 110), panelRect.width - 110);
      let top = rect.top - panelRect.top - 44;
      if (top < 64) top = rect.bottom - panelRect.top + 8;
      setSelectionAction({ text, left, top });
    });
  }

  function resetChat() {
    chatAbortRef.current?.abort();
    chatAbortRef.current = null;
    setChatMessages([]);
    setConversations([]);
    setActiveConversationId("");
    setChatInput("");
    setChatQuote("");
    setChatStreaming(false);
    setConversationLoading(false);
  }

  async function openComparisonChat(quote = "") {
    if (!comparisonId) return;
    setChatQuote(quote);
    setChatOpen(true);
    setSelectionAction(null);
    window.getSelection()?.removeAllRanges();
    await loadConversations(activeConversationId).catch((chatError) => {
      showToast(chatError instanceof Error ? chatError.message : "无法读取对比追问记录。" );
    });
  }

  async function loadConversations(preferredId = "") {
    if (!comparisonId) return [];
    setConversationLoading(true);
    try {
      const response = await fetch(`/api/comparisons/${encodeURIComponent(comparisonId)}/conversations`);
      if (!response.ok) throw new Error(`无法读取对话（HTTP ${response.status}）`);
      const payload = await response.json();
      const items = Array.isArray(payload.items) ? payload.items : [];
      setConversations(items);
      const target = items.some((item) => item.id === preferredId) ? preferredId : items[0]?.id || "";
      if (target) await loadConversation(target, false);
      else {
        setActiveConversationId("");
        setChatMessages([]);
      }
      return items;
    } finally {
      setConversationLoading(false);
    }
  }

  async function loadConversation(conversationId, manageLoading = true) {
    if (!conversationId) {
      setActiveConversationId("");
      setChatMessages([]);
      return;
    }
    if (manageLoading) setConversationLoading(true);
    try {
      const response = await fetch(`/api/comparisons/chat/conversations/${encodeURIComponent(conversationId)}`);
      if (!response.ok) throw new Error(`无法恢复对话（HTTP ${response.status}）`);
      const payload = await response.json();
      setActiveConversationId(payload.conversation?.id || conversationId);
      setChatMessages(Array.isArray(payload.messages) ? payload.messages : []);
    } finally {
      if (manageLoading) setConversationLoading(false);
    }
  }

  function startNewConversation() {
    chatAbortRef.current?.abort();
    setActiveConversationId("");
    setChatMessages([]);
    setChatInput("");
    setChatStreaming(false);
  }

  async function selectConversation(conversationId) {
    if (!conversationId) return startNewConversation();
    try {
      await loadConversation(conversationId);
    } catch (selectError) {
      showToast(selectError instanceof Error ? selectError.message : "无法恢复对话。" );
    }
  }

  async function renameConversation(title) {
    if (!activeConversationId) return false;
    setConversationLoading(true);
    try {
      const response = await fetch(`/api/comparisons/chat/conversations/${encodeURIComponent(activeConversationId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!response.ok) throw new Error(`无法修改会话名称（HTTP ${response.status}）`);
      const payload = await response.json();
      setConversations((previous) => previous.map((item) => (
        item.id === payload.conversation.id ? payload.conversation : item
      )));
      showToast("会话名称已更新");
      return true;
    } catch (renameError) {
      showToast(renameError instanceof Error ? renameError.message : "无法修改会话名称。" );
      return false;
    } finally {
      setConversationLoading(false);
    }
  }

  async function deleteConversation() {
    if (!activeConversationId) return;
    const deletingId = activeConversationId;
    setConversationLoading(true);
    try {
      const response = await fetch(`/api/comparisons/chat/conversations/${encodeURIComponent(deletingId)}`, { method: "DELETE" });
      if (!response.ok) throw new Error(`无法删除对话（HTTP ${response.status}）`);
      const remaining = conversations.filter((item) => item.id !== deletingId);
      setConversations(remaining);
      if (remaining[0]?.id) await loadConversation(remaining[0].id, false);
      else {
        setActiveConversationId("");
        setChatMessages([]);
      }
      showToast("对话已删除");
    } catch (deleteError) {
      showToast(deleteError instanceof Error ? deleteError.message : "无法删除对话。" );
    } finally {
      setConversationLoading(false);
    }
  }

  async function sendChatMessage() {
    const question = chatInput.trim();
    if (!question || chatStreaming || !comparisonId) return;
    const quote = chatQuote.trim();
    const userId = `compare-user-${Date.now()}`;
    const assistantId = `compare-assistant-${Date.now()}`;
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
      const response = await fetch(`/api/comparisons/chat/stream?demo=${data?.mode === "demo" ? "true" : "false"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          comparison_id: comparisonId,
          conversation_id: activeConversationId || null,
          question,
          selected_text: quote || null,
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `跨论文追问失败（HTTP ${response.status}）`);
      }
      let answer = "";
      let complete = null;
      await readNdjson(response.body, (event) => {
        if (event.type === "error") throw new Error(event.message || "跨论文追问失败。");
        if (event.type === "token") {
          answer += event.text || "";
          setChatMessages((previous) => previous.map((message) => (
            message.id === assistantId ? { ...message, content: answer } : message
          )));
        }
        if (event.type === "complete") complete = event;
      });
      if (!complete || !answer.trim()) throw new Error("回答在完成前意外结束。");
      setChatMessages((previous) => previous.map((message) => (
        message.id === userId && complete.user_message
          ? complete.user_message
          : message.id === assistantId
            ? complete.assistant_message || { ...message, content: answer }
            : message
      )));
      setActiveConversationId(complete.conversation_id || "");
      if (complete.conversation) {
        setConversations((previous) => [
          complete.conversation,
          ...previous.filter((item) => item.id !== complete.conversation.id),
        ]);
      }
    } catch (chatError) {
      if (chatError?.name === "AbortError") return;
      const message = chatError instanceof Error ? chatError.message : "跨论文追问失败。";
      setChatMessages((previous) => previous.map((item) => (
        item.id === assistantId ? { ...item, content: message, error: true } : item
      )));
    } finally {
      if (chatAbortRef.current === controller) chatAbortRef.current = null;
      setChatStreaming(false);
    }
  }

  return (
    <main className="comparison-workspace">
      <aside className="comparison-sidebar glass">
        <section className="comparison-picker">
          <header>
            <div>
              <span>对比论文</span>
              <small>{selectedIds.length} / 4</small>
            </div>
            <button type="button" title="分析新论文" aria-label="添加并分析新论文" onClick={onAddPaper}>
              <IconPlus size={18} stroke={1.9} />
            </button>
          </header>
          <div className="comparison-paper-list">
            {historyLoading && <div className="comparison-empty-row"><IconLoader2 className="spin" size={18} /> 正在读取论文</div>}
            {!historyLoading && historyError && <div className="comparison-empty-row error">{historyError}</div>}
            {!historyLoading && !historyError && !historyItems.length && (
              <div className="comparison-empty-row">暂无已分析论文</div>
            )}
            {historyItems.map((paper) => {
              const checked = selectedIds.includes(paper.id);
              return (
                <label className={`comparison-paper-option ${checked ? "selected" : ""}`} key={paper.id}>
                  <input type="checkbox" checked={checked} onChange={() => togglePaper(paper.id)} />
                  <span className="comparison-check">{checked && <IconCheck size={13} stroke={2.4} />}</span>
                  <IconFileTypePdf className="comparison-pdf-icon" size={22} stroke={1.7} />
                  <span className="comparison-paper-copy">
                    <strong>{paper.title}</strong>
                    <small>{paper.pages || "—"} 页 · {paper.filename}</small>
                  </span>
                  {checked && <b>{paperLabels[paper.id]}</b>}
                </label>
              );
            })}
          </div>
        </section>

        <section className="comparison-focus">
          <span>对比重点</span>
          <div className="focus-segments" role="group" aria-label="选择对比重点">
            {focusOptions.map((option) => (
              <button
                className={focus === option.value ? "active" : ""}
                type="button"
                key={option.value}
                onClick={() => setFocus(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
          {focus === "custom" && (
            <textarea
              rows={3}
              maxLength={1000}
              value={customFocus}
              placeholder="输入需要回答的跨论文问题"
              onChange={(event) => setCustomFocus(event.target.value)}
            />
          )}
          <button
            className="run-comparison"
            type="button"
            disabled={selectedIds.length < 2 || isComparing || (focus === "custom" && !customFocus.trim())}
            onClick={startComparison}
          >
            {isComparing ? <IconLoader2 className="spin" size={18} /> : <IconSparkles size={18} />}
            {isComparing ? "正在比较" : "开始对比"}
          </button>
        </section>

        <section className={`saved-comparisons ${savedOpen ? "expanded" : "collapsed"}`}>
          <button className="saved-comparisons-toggle" type="button" onClick={() => setSavedOpen((value) => !value)}>
            <span><IconHistory size={16} /> 已保存对比</span>
            <IconChevronRight className="saved-comparisons-chevron" size={16} />
          </button>
          <div className="saved-comparisons-list">
            {savedLoading && <div className="comparison-empty-row"><IconLoader2 className="spin" size={16} /> 正在读取</div>}
            {!savedLoading && !savedItems.length && <div className="comparison-empty-row">暂无对比记录</div>}
            {savedItems.map((item) => (
              <div className={`saved-comparison-row ${comparisonId === item.id ? "active" : ""}`} key={item.id}>
                <button type="button" disabled={busyId === item.id} onClick={() => openSavedComparison(item)}>
                  <strong>{item.title}</strong>
                  <small>{item.paper_count} 篇 · {formatAge(item.updated_at)}</small>
                </button>
                <button
                  className="saved-comparison-delete"
                  type="button"
                  aria-label={`删除 ${item.title}`}
                  title="删除对比记录"
                  disabled={busyId === item.id}
                  onClick={() => deleteSavedComparison(item)}
                >
                  <IconTrash size={14} />
                </button>
              </div>
            ))}
          </div>
        </section>
      </aside>

      <section className="comparison-result-panel glass" ref={resultPanelRef}>
        {!comparison && !isComparing && !error && (
          <div className="comparison-welcome">
            <span><IconArrowsLeftRight size={30} stroke={1.45} /></span>
            <strong>选择 2 至 4 篇论文</strong>
            <p>多论文对比</p>
          </div>
        )}
        {isComparing && (
          <div className="comparison-running">
            <IconLoader2 className="spin" size={28} stroke={1.5} />
            <strong>{progress}</strong>
            <div className="loaded-paper-dots">
              {selectedIds.map((id, index) => (
                <span className={loadedLabels.includes(`P${index + 1}`) ? "complete" : ""} key={id}>P{index + 1}</span>
              ))}
            </div>
          </div>
        )}
        {error && !isComparing && (
          <div className="comparison-error">
            <IconAlertCircle size={28} />
            <strong>对比未完成</strong>
            <p>{error}</p>
          </div>
        )}
        {comparison && (
          <>
            <header className="comparison-result-header">
              <div>
                <small>{comparison.focus}</small>
                <h1>{comparison.title}</h1>
              </div>
              <div className="comparison-coverage" title="根据带有效原文证据的论文维度占比计算">
                <strong>{assessment?.evidence_coverage ?? 0}%</strong>
                <span>证据覆盖</span>
              </div>
            </header>
            <div className="comparison-paper-legend">
              {comparison.papers.map((paper) => (
                <span key={paper.label}><b>{paper.label}</b><strong>{paper.title}</strong></span>
              ))}
            </div>
            <div className="comparison-tabs" role="tablist">
              {resultTabs.map((tab) => (
                <button
                  className={activeTab === tab.value ? "active" : ""}
                  type="button"
                  role="tab"
                  aria-selected={activeTab === tab.value}
                  key={tab.value}
                  onClick={() => { setActiveTab(tab.value); setSelectionAction(null); }}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <div
              className="comparison-result-scroll"
              ref={resultScrollRef}
              onMouseUp={handleResultSelection}
              onKeyUp={handleResultSelection}
              onScroll={() => setSelectionAction(null)}
            >
              {activeTab === "overview" && (
                <ComparisonOverview comparison={comparison} />
              )}
              {activeTab !== "gaps" && (
                <ComparisonMatrix
                  comparison={comparison}
                  category={activeTab === "overview" ? null : activeTab}
                  onEvidence={(evidenceId) => setSelectedEvidence(evidenceById[evidenceId] || { id: evidenceId })}
                />
              )}
              {activeTab === "gaps" && (
                <ComparisonGaps comparison={comparison} assessment={assessment} />
              )}
            </div>
            {selectionAction && !chatOpen && (
              <button
                className="selection-chat-action comparison-selection-action"
                type="button"
                style={{ left: selectionAction.left, top: selectionAction.top }}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => openComparisonChat(selectionAction.text)}
              >
                <IconMessageCircle size={16} /> 在对比追问中提问
              </button>
            )}
            {selectedEvidence && (
              <aside className="comparison-evidence-panel">
                <header>
                  <div><b>{selectedEvidence.id}</b><span>{selectedEvidence.page_label || "证据"}</span></div>
                  <button type="button" aria-label="关闭证据" onClick={() => setSelectedEvidence(null)}><IconX size={17} /></button>
                </header>
                <strong>{selectedEvidence.paper_title || selectedEvidence.paper_label}</strong>
                <small>{selectedEvidence.section}</small>
                <p>{selectedEvidence.preview || "该证据预览未保存在当前结果中。"}</p>
              </aside>
            )}
            {!chatOpen && (
              <button
                className="open-comparison-chat"
                type="button"
                aria-label="打开跨论文追问"
                title="跨论文追问"
                onClick={() => openComparisonChat("")}
              >
                <IconSparkles size={19} />
              </button>
            )}
            {chatOpen && (
              <ComparisonChatDrawer
                comparisonTitle={comparison.title}
                paperCount={comparison.papers.length}
                conversations={conversations}
                activeConversationId={activeConversationId}
                messages={chatMessages}
                input={chatInput}
                quote={chatQuote}
                isStreaming={chatStreaming}
                isConversationLoading={conversationLoading}
                onInputChange={setChatInput}
                onClearQuote={() => setChatQuote("")}
                onSend={sendChatMessage}
                onClose={() => setChatOpen(false)}
                onConversationChange={selectConversation}
                onNewConversation={startNewConversation}
                onDeleteConversation={deleteConversation}
                onRenameConversation={renameConversation}
              />
            )}
          </>
        )}
      </section>
    </main>
  );
}

function ComparisonOverview({ comparison }) {
  return (
    <div className="comparison-overview">
      <section>
        <span>综合结论</span>
        <p>{comparison.executive_summary}</p>
      </section>
      <div className="comparison-overview-columns">
        <section>
          <span>共同基础</span>
          <ul>{comparison.common_ground.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>
        </section>
        <section>
          <span>关键差异</span>
          <ul>{comparison.key_differences.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>
        </section>
      </div>
    </div>
  );
}

function ComparisonMatrix({ comparison, category, onEvidence }) {
  const dimensions = category
    ? comparison.dimensions.filter((dimension) => dimension.category === category)
    : comparison.dimensions;
  if (!dimensions.length) return <div className="comparison-no-dimensions">当前结果没有该类别的对比维度。</div>;
  return (
    <div className="comparison-matrix-scroll">
      <table className="comparison-matrix" style={{ "--paper-count": comparison.papers.length }}>
        <thead>
          <tr>
            <th>对比维度</th>
            {comparison.papers.map((paper) => <th key={paper.label}><b>{paper.label}</b>{paper.title}</th>)}
            <th>跨论文结论</th>
          </tr>
        </thead>
        <tbody>
          {dimensions.map((dimension) => {
            const cells = Object.fromEntries(dimension.cells.map((cell) => [cell.paper_label, cell]));
            return (
              <tr key={dimension.key}>
                <th>
                  <strong>{dimension.title}</strong>
                  <small>{dimension.description}</small>
                </th>
                {comparison.papers.map((paper) => {
                  const cell = cells[paper.label];
                  return (
                    <td key={paper.label}>
                      <p>{cell?.summary || "证据不足"}</p>
                      {!!cell?.evidence_ids?.length && (
                        <div className="comparison-citations">
                          {cell.evidence_ids.map((evidenceId) => (
                            <button type="button" key={evidenceId} onClick={() => onEvidence(evidenceId)}>{evidenceId}</button>
                          ))}
                        </div>
                      )}
                    </td>
                  );
                })}
                <td className="comparison-synthesis-cell">
                  <span className={`comparability ${dimension.comparability}`}>
                    {comparabilityLabels[dimension.comparability]}
                  </span>
                  <p>{dimension.synthesis}</p>
                  {dimension.warning && <small>{dimension.warning}</small>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonGaps({ comparison, assessment }) {
  const sections = [
    ["研究空白", comparison.research_gaps],
    ["适用建议", comparison.recommendations],
    ["注意事项", [...(comparison.warnings || []), ...(assessment?.warnings || [])]],
  ];
  return (
    <div className="comparison-gaps">
      {sections.map(([title, items]) => (
        <section key={title}>
          <span>{title}</span>
          {items?.length ? <ul>{items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul> : <p>暂无相关内容。</p>}
        </section>
      ))}
    </div>
  );
}

function ComparisonChatDrawer({
  comparisonTitle,
  paperCount,
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
  const messagesEndRef = useRef(null);
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const activeConversation = conversations.find((item) => item.id === activeConversationId);

  useEffect(() => { textareaRef.current?.focus(); }, [quote]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ block: "end" }); }, [messages, isStreaming]);
  useEffect(() => {
    setIsRenaming(false);
    setRenameValue(activeConversation?.title || "");
  }, [activeConversationId, activeConversation?.title]);

  function handleInputKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (input.trim() && !isStreaming) onSend();
    }
  }

  async function submitRename() {
    if (!renameValue.trim()) return;
    const saved = await onRenameConversation(renameValue.trim());
    if (saved !== false) setIsRenaming(false);
  }

  return (
    <section className="paper-chat-drawer comparison-chat-drawer" aria-label="跨论文追问">
      <header className="chat-header">
        <div>
          <span><IconMessageCircle size={16} /> 对比追问</span>
          <strong title={comparisonTitle}>{paperCount} 篇论文 · {comparisonTitle}</strong>
        </div>
        <div className="chat-header-actions">
          <button type="button" title="删除当前对话" disabled={!activeConversationId || isStreaming || isConversationLoading} onClick={onDeleteConversation}><IconTrash size={17} /></button>
          <button type="button" title="关闭追问" onClick={onClose}><IconX size={18} /></button>
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
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.nativeEvent.isComposing) void submitRename();
                if (event.key === "Escape") setIsRenaming(false);
              }}
            />
            <button type="button" title="保存名称" disabled={!renameValue.trim()} onClick={() => void submitRename()}><IconCheck size={17} /></button>
            <button type="button" title="取消编辑" onClick={() => setIsRenaming(false)}><IconX size={17} /></button>
          </>
        ) : (
          <>
            <select value={activeConversationId || ""} onChange={(event) => onConversationChange(event.target.value)} disabled={isStreaming || isConversationLoading}>
              <option value="">新对话</option>
              {conversations.map((conversation) => (
                <option value={conversation.id} key={conversation.id}>{conversation.title}（{conversation.message_count}）</option>
              ))}
            </select>
            <button type="button" title="修改当前会话名称" disabled={!activeConversationId || isStreaming} onClick={() => setIsRenaming(true)}><IconPencil size={16} /></button>
            <button type="button" title="新建对话" disabled={isStreaming} onClick={onNewConversation}><IconPlus size={17} /></button>
          </>
        )}
      </div>
      <div className="chat-messages">
        {isConversationLoading && <div className="chat-empty"><IconLoader2 className="spin" size={22} /><strong>正在恢复对话</strong></div>}
        {!isConversationLoading && !messages.length && <div className="chat-empty"><IconArrowsLeftRight size={24} /><strong>GLM-5.2</strong></div>}
        {messages.map((message) => (
          <article className={`chat-message ${message.role}${message.error ? " error" : ""}`} key={message.id}>
            {message.quote && <blockquote><IconQuote size={14} /> {message.quote}</blockquote>}
            {message.content ? (
              message.role === "assistant" ? (
                <ReactMarkdown components={chatMarkdownComponents} remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
              ) : <p>{message.content}</p>
            ) : <span className="chat-typing"><i /><i /><i /></span>}
          </article>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form className="chat-composer" onSubmit={(event) => { event.preventDefault(); onSend(); }}>
        {quote && (
          <div className="chat-quote-chip"><IconQuote size={14} /><span>{quote}</span><button type="button" onClick={onClearQuote}><IconX size={14} /></button></div>
        )}
        <div className="chat-input-row">
          <textarea ref={textareaRef} value={input} maxLength={4000} rows={2} placeholder="继续追问这些论文" onChange={(event) => onInputChange(event.target.value)} onKeyDown={handleInputKeyDown} />
          <button className="chat-send" type="submit" disabled={!input.trim() || isStreaming}>{isStreaming ? <IconLoader2 className="spin" size={18} /> : <IconSend size={18} />}</button>
        </div>
      </form>
    </section>
  );
}

async function readNdjson(body, onEvent) {
  if (!body) throw new Error("响应没有可读取的数据流。");
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}
