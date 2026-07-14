import {
  IconAlertTriangle,
  IconRoute,
  IconShieldCheck,
} from "@tabler/icons-react";

export function ModelCallTrace({ trace }) {
  if (!trace?.provider || !trace?.requested_model) return null;
  const verification = trace.verification || "route_recorded";
  const isMismatch = verification === "upstream_mismatch";
  const isConfirmed = verification === "upstream_confirmed";
  const modelLabel = trace.requested_model_label || trace.requested_model;

  return (
    <div
      className={`model-call-trace ${isMismatch ? "warning" : isConfirmed ? "confirmed" : "recorded"}`}
      aria-label={modelLabel}
    >
      {isMismatch ? (
        <IconAlertTriangle size={14} stroke={2} />
      ) : isConfirmed ? (
        <IconShieldCheck size={14} stroke={2} />
      ) : (
        <IconRoute size={14} stroke={2} />
      )}
      <span>{modelLabel}</span>
    </div>
  );
}
