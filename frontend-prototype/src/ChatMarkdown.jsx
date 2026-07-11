import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";


const markdownComponents = {
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer">{children}</a>
  ),
  table: ({ children }) => (
    <div className="chat-table-scroll">
      <table>{children}</table>
    </div>
  ),
};


export function ChatMarkdown({ children }) {
  return (
    <ReactMarkdown
      components={markdownComponents}
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
    >
      {normalizeMathDelimiters(children)}
    </ReactMarkdown>
  );
}


export function normalizeMathDelimiters(markdown) {
  return String(markdown || "")
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((part, index) => {
      if (index % 2 === 1) return part;
      return part
        .replace(/\\\[([\s\S]*?)\\\]/g, (_, formula) => `\n$$\n${formula.trim()}\n$$\n`)
        .replace(/\\\(([^\n]*?)\\\)/g, (_, formula) => `$${formula.trim()}$`);
    })
    .join("");
}
