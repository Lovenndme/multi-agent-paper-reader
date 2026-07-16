import assert from "node:assert/strict";
import test from "node:test";

import {
  applyAnalysisProcessEvent,
  emptyAnalysisProcess,
  formatAnalysisDuration,
  normalizeAnalysisProcess,
} from "./analysisProcess.js";

test("builds a readable process trace without structured tokens", () => {
  let process = applyAnalysisProcessEvent(emptyAnalysisProcess(), {
    type: "analysis_started",
    started_at: "2026-07-17T00:00:00Z",
  });
  process = applyAnalysisProcessEvent(process, {
    type: "agent_started",
    agent: "method",
    summary: "正在识别方法组件。",
    elapsed_ms: 1000,
  });
  process = applyAnalysisProcessEvent(process, {
    type: "agent_progress",
    agent: "method",
    progress_id: "reasoning-0",
    text: "正在核对组件之间的关系。",
    source: "native_reasoning_summary",
    elapsed_ms: 2200,
  });
  process = applyAnalysisProcessEvent(process, {
    type: "agent_complete",
    agent: "method",
    summary: "方法分析已完成。",
    duration_ms: 3100,
    elapsed_ms: 3200,
  });

  assert.equal(process.agents.method.status, "complete");
  assert.equal(process.agents.method.duration_ms, 3100);
  assert.equal(process.entries.at(-2).text, "正在核对组件之间的关系。");
  assert.equal(process.entries.at(-1).text, "方法分析已完成。");
});

test("replaces incremental native reasoning summaries instead of duplicating them", () => {
  let process = emptyAnalysisProcess();
  process = applyAnalysisProcessEvent(process, {
    type: "agent_progress",
    agent: "critic",
    progress_id: "reasoning-0",
    text: "正在核对",
  });
  process = applyAnalysisProcessEvent(process, {
    type: "agent_progress",
    agent: "critic",
    progress_id: "reasoning-0",
    text: "正在核对实验结论。",
  });

  assert.equal(process.entries.length, 1);
  assert.equal(process.entries[0].text, "正在核对实验结论。");
});

test("normalizes stored history and formats elapsed time", () => {
  const restored = normalizeAnalysisProcess({
    status: "completed",
    duration_ms: 350000,
    entries: [{ id: "summary-complete", agent: "summary", text: "最终笔记已完成。" }],
  });

  assert.equal(restored.status, "completed");
  assert.equal(formatAnalysisDuration(restored.duration_ms), "5分50秒");
  assert.equal(formatAnalysisDuration(9000), "9秒");
});
