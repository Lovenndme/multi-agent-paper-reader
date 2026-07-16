export const analysisAgentOrder = ["system", "method", "experiment", "critic", "summary"];

export function emptyAnalysisProcess() {
  return {
    status: "idle",
    started_at: null,
    completed_at: null,
    duration_ms: 0,
    agents: {},
    entries: [],
  };
}

export function normalizeAnalysisProcess(value) {
  if (!value || typeof value !== "object") return emptyAnalysisProcess();
  return {
    status: value.status || "completed",
    started_at: value.started_at || null,
    completed_at: value.completed_at || null,
    duration_ms: positiveNumber(value.duration_ms),
    agents: value.agents && typeof value.agents === "object" ? value.agents : {},
    entries: Array.isArray(value.entries)
      ? value.entries.filter((entry) => entry && typeof entry === "object" && entry.text)
      : [],
  };
}

export function applyAnalysisProcessEvent(current, event) {
  const process = normalizeAnalysisProcess(current);
  if (!event || typeof event !== "object") return process;

  if (event.type === "analysis_started") {
    return {
      ...emptyAnalysisProcess(),
      status: "running",
      started_at: event.started_at || new Date().toISOString(),
      duration_ms: positiveNumber(event.elapsed_ms),
    };
  }

  if (event.type === "agent_started") {
    return {
      ...process,
      status: "running",
      duration_ms: positiveNumber(event.elapsed_ms),
      agents: {
        ...process.agents,
        [event.agent]: {
          status: "running",
          started_at: event.started_at || null,
          duration_ms: 0,
        },
      },
      entries: upsertEntry(process.entries, {
        id: `${event.agent}-start`,
        agent: event.agent,
        text: event.summary,
        source: event.source || "pipeline",
        elapsed_ms: positiveNumber(event.elapsed_ms),
      }),
    };
  }

  if (event.type === "agent_progress") {
    return {
      ...process,
      status: "running",
      duration_ms: positiveNumber(event.elapsed_ms),
      entries: upsertEntry(process.entries, {
        id: event.progress_id,
        agent: event.agent || "system",
        text: event.text,
        source: event.source || "pipeline",
        elapsed_ms: positiveNumber(event.elapsed_ms),
      }),
    };
  }

  if (event.type === "agent_complete") {
    return {
      ...process,
      duration_ms: positiveNumber(event.elapsed_ms),
      agents: {
        ...process.agents,
        [event.agent]: {
          ...(process.agents[event.agent] || {}),
          status: "complete",
          completed_at: event.completed_at || null,
          duration_ms: positiveNumber(event.duration_ms),
        },
      },
      entries: upsertEntry(process.entries, {
        id: `${event.agent}-complete`,
        agent: event.agent,
        text: event.summary,
        source: event.source || "pipeline",
        elapsed_ms: positiveNumber(event.elapsed_ms),
      }),
    };
  }

  if (event.type === "complete" && event.analysis_process) {
    return normalizeAnalysisProcess(event.analysis_process);
  }

  if (event.type === "error") {
    if (event.analysis_process) return normalizeAnalysisProcess(event.analysis_process);
    return {
      ...process,
      status: "failed",
      duration_ms: positiveNumber(event.elapsed_ms || process.duration_ms),
      agents: event.agent
        ? {
            ...process.agents,
            [event.agent]: {
              ...(process.agents[event.agent] || {}),
              status: "failed",
              duration_ms: positiveNumber(event.duration_ms),
            },
          }
        : process.agents,
    };
  }

  return process;
}

export function formatAnalysisDuration(value) {
  const totalSeconds = Math.max(0, Math.floor(positiveNumber(value) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours) return `${hours}小时${minutes}分${seconds}秒`;
  if (minutes) return `${minutes}分${seconds}秒`;
  return `${seconds}秒`;
}

function upsertEntry(entries, next) {
  if (!next.id || !next.text) return entries;
  const index = entries.findIndex((entry) => entry.id === next.id && entry.agent === next.agent);
  if (index < 0) return [...entries, next].slice(-40);
  const updated = [...entries];
  updated[index] = { ...updated[index], ...next };
  return updated;
}

function positiveNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : 0;
}
