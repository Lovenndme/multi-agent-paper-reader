const metricGroups = [
  {
    pattern: /BLEU-1\/2\/3\/4\s*=\s*([\d.]+)\/([\d.]+)\/([\d.]+)\/([\d.]+)/i,
    keys: ["bleu1", "bleu2", "bleu3", "bleu4"],
  },
  {
    pattern: /ROUGE-F\/P\/R\s*=\s*([\d.]+)\/([\d.]+)\/([\d.]+)/i,
    keys: ["rougeF", "rougeP", "rougeR"],
  },
  {
    pattern: /BERTScore-F\s*=\s*([\d.]+)/i,
    keys: ["bertScoreF"],
  },
];

export const experimentMetricColumns = [
  ["setting", "序列长度"],
  ["bleu1", "BLEU-1"],
  ["bleu2", "BLEU-2"],
  ["bleu3", "BLEU-3"],
  ["bleu4", "BLEU-4"],
  ["rougeF", "ROUGE-F"],
  ["rougeP", "ROUGE-P"],
  ["rougeR", "ROUGE-R"],
  ["bertScoreF", "BERTScore-F"],
];

export function parseExperimentMetricTable(text = "") {
  const rows = [];
  const settingPattern = /(\d+\s*TR)\s*时(?:进一步)?达到\s*([^；。]+)/gi;
  for (const match of text.matchAll(settingPattern)) {
    const row = { setting: match[1].replace(/\s+/g, "") };
    for (const group of metricGroups) {
      const values = match[2].match(group.pattern);
      if (!values) continue;
      group.keys.forEach((key, index) => {
        row[key] = values[index + 1];
      });
    }
    if (Object.keys(row).length === 1) {
      const compactValues = match[2].match(
        /([\d.]+)\/([\d.]+)\/([\d.]+)\/([\d.]+)[、,]\s*([\d.]+)\/([\d.]+)\/([\d.]+)\s*(?:和|、)\s*([\d.]+)/,
      );
      if (compactValues) {
        experimentMetricColumns.slice(1).forEach(([key], index) => {
          row[key] = compactValues[index + 1];
        });
      }
    }
    if (experimentMetricColumns.slice(1).every(([key]) => row[key] !== undefined)) {
      rows.push(row);
    }
  }
  return rows.length >= 2 ? rows : [];
}
