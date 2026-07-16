import assert from "node:assert/strict";
import test from "node:test";

import { parseExperimentMetricTable } from "./experimentTable.js";


test("parses repeated TR metric groups from an experiment summary", () => {
  const rows = parseExperimentMetricTable(
    "20TR 时达到 BLEU-1/2/3/4=25.4/10.5/4.7/2.6、"
    + "ROUGE-F/P/R=23.4/22.6/24.6、BERTScore-F=46.3；"
    + "40TR 时达到 31.2/15.3/10.3/8.2、29.6/28.7/30.4 和 50.0；"
    + "60TR 时进一步达到 36.2/20.4/14.7/12.1、36.2/35.6/37.2 和 53.5。",
  );

  assert.equal(rows.length, 3);
  assert.deepEqual(rows[0], {
    setting: "20TR",
    bleu1: "25.4",
    bleu2: "10.5",
    bleu3: "4.7",
    bleu4: "2.6",
    rougeF: "23.4",
    rougeP: "22.6",
    rougeR: "24.6",
    bertScoreF: "46.3",
  });
  assert.equal(rows[2].bertScoreF, "53.5");
});


test("does not invent a table for ordinary prose", () => {
  assert.deepEqual(parseExperimentMetricTable("结果整体优于基线，但未提供完整矩阵。"), []);
});
