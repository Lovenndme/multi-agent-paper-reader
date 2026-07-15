import {
  IconAlertTriangle,
  IconRoute,
  IconSearch,
  IconShieldCheck,
  IconUsers,
} from "@tabler/icons-react";


const effortLabels = {
  low: "轻度",
  medium: "中等",
  high: "高",
  xhigh: "最高",
  max: "极高",
  ultra: "Ultra",
};


export function ModelCallTrace({ trace }) {
  if (!trace?.provider || !trace?.requested_model) return null;
  const verification = trace.verification || "route_recorded";
  const isMismatch = verification === "upstream_mismatch";
  const isConfirmed = verification === "upstream_confirmed";
  const modelLabel = trace.requested_model_label || trace.requested_model;
  const effort = String(trace.effort || "").toLowerCase();
  const tools = Array.isArray(trace.tools_used)
    ? trace.tools_used.map((tool) => String(tool)).filter(Boolean).slice(0, 12)
    : [];
  const subagentCount = Math.max(0, Number.parseInt(trace.subagent_count, 10) || 0);

  return (
    <div
      className={`model-call-trace ${isMismatch ? "warning" : isConfirmed ? "confirmed" : "recorded"}`}
      aria-label={`模型调用详情：${modelLabel}`}
    >
      <div className="model-call-identity">
        {isMismatch ? (
          <IconAlertTriangle size={14} stroke={2} />
        ) : isConfirmed ? (
          <IconShieldCheck size={14} stroke={2} />
        ) : (
          <IconRoute size={14} stroke={2} />
        )}
        <span>{modelLabel}</span>
      </div>
      {(effort || trace.web_search_used || tools.length || subagentCount > 0) && (
        <div className="model-call-capabilities">
          {effort && <span className={effort === "ultra" ? "ultra" : ""}>{effortLabels[effort] || effort}</span>}
          {trace.web_search_used && <span><IconSearch size={11} /> Web Search</span>}
          {tools.length > 0 && (
            <span title={tools.join("、")}>工具 {tools.length}</span>
          )}
          {subagentCount > 0 && <span className="ultra"><IconUsers size={11} /> 子 Agent {subagentCount}</span>}
        </div>
      )}
    </div>
  );
}
