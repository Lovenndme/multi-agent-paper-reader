import {
  IconAlertTriangle,
  IconChevronDown,
  IconRoute,
  IconShieldCheck,
} from "@tabler/icons-react";

const verificationLabels = {
  upstream_confirmed: "上游已确认",
  upstream_mismatch: "响应模型不一致",
  endpoint_confirmed: "厂商端点已响应",
  route_recorded: "后端路由已记录",
};

function TraceRow({ label, value }) {
  if (!value) return null;
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export function ModelCallTrace({ trace }) {
  if (!trace?.provider || !trace?.requested_model) return null;
  const verification = trace.verification || "route_recorded";
  const isMismatch = verification === "upstream_mismatch";
  const isConfirmed = verification === "upstream_confirmed";
  const modelLabel = trace.requested_model_label || trace.requested_model;
  const summary = `${trace.provider_label || trace.provider} · ${modelLabel}`;

  return (
    <details className={`model-call-trace ${isMismatch ? "warning" : isConfirmed ? "confirmed" : "recorded"}`}>
      <summary>
        {isMismatch ? (
          <IconAlertTriangle size={14} stroke={2} />
        ) : isConfirmed ? (
          <IconShieldCheck size={14} stroke={2} />
        ) : (
          <IconRoute size={14} stroke={2} />
        )}
        <span>{summary}</span>
        <em>{verificationLabels[verification] || verificationLabels.route_recorded}</em>
        <IconChevronDown className="model-call-trace-chevron" size={13} stroke={2} />
      </summary>
      <dl>
        <TraceRow label="请求路由" value={summary} />
        <TraceRow label="厂商端点" value={trace.endpoint_host} />
        <TraceRow label="上游响应模型" value={trace.upstream_model} />
        <TraceRow label="厂商请求 ID" value={trace.request_id} />
      </dl>
    </details>
  );
}
