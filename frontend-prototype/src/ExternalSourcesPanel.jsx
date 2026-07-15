import {
  IconBook2,
  IconExternalLink,
  IconWorldSearch,
} from "@tabler/icons-react";


function safeHttpUrl(value) {
  if (!value) return "";
  try {
    const url = new URL(String(value));
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : "";
  } catch {
    return "";
  }
}


function sourceDomain(source, url) {
  const supplied = String(source?.domain || "").trim();
  if (supplied) return supplied;
  if (!url) return "外部资料";
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "外部资料";
  }
}


export function collectExternalSources(message) {
  const candidates = [
    ...(Array.isArray(message?.external_sources) ? message.external_sources : []),
    ...(Array.isArray(message?.model_trace?.external_sources) ? message.model_trace.external_sources : []),
  ];
  const seen = new Set();
  return candidates.flatMap((source, index) => {
    if (!source || typeof source !== "object") return [];
    const title = String(source.title || source.name || "外部资料").trim().slice(0, 240);
    const url = safeHttpUrl(source.url);
    const key = url || `${title}:${source.id || index}`;
    if (seen.has(key)) return [];
    seen.add(key);
    return [{
      id: String(source.id || `S${index + 1}`).slice(0, 32),
      title,
      url,
      domain: sourceDomain(source, url).slice(0, 120),
      sourceType: String(source.source_type || "web_search").slice(0, 48),
    }];
  }).slice(0, 12);
}


export function ExternalSourcesPanel({ message, sources }) {
  const normalized = collectExternalSources({
    external_sources: Array.isArray(sources) ? sources : message?.external_sources,
    model_trace: message?.model_trace,
  });
  if (!normalized.length) return null;

  return (
    <section className="external-sources-panel" aria-label="外部资料来源">
      <header>
        <span><IconWorldSearch size={14} stroke={1.9} /> 外部资料</span>
        <small>{normalized.length} 个来源</small>
      </header>
      <div className="external-source-list">
        {normalized.map((source) => (
          <article key={`${source.id}:${source.url || source.title}`}>
            <span className="external-source-icon" aria-hidden="true">
              {source.sourceType === "web_search"
                ? <IconWorldSearch size={14} stroke={1.8} />
                : <IconBook2 size={14} stroke={1.8} />}
            </span>
            <div>
              {source.url ? (
                <a href={source.url} target="_blank" rel="noopener noreferrer">
                  {source.title} <IconExternalLink size={12} stroke={1.8} />
                </a>
              ) : <strong>{source.title}</strong>}
              <small>{source.id} · {source.domain}</small>
            </div>
          </article>
        ))}
      </div>
      <p>这些来源来自联网检索，用于补充背景，不替代当前论文原文证据。</p>
    </section>
  );
}
