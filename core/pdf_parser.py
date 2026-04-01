"""PDF parsing and section splitting utilities."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF


# Common section header patterns in academic papers
SECTION_PATTERNS = [
    r"^(abstract)$",
    r"^(\d+\.?\s+introduction)$",
    r"^(\d+\.?\s+related\s+work)$",
    r"^(\d+\.?\s+background)$",
    r"^(\d+\.?\s+method(?:ology)?)$",
    r"^(\d+\.?\s+(?:proposed\s+)?(?:model|approach|framework|system|architecture))$",
    r"^(\d+\.?\s+experiment(?:s|al\s+(?:setup|results|evaluation))?)$",
    r"^(\d+\.?\s+result(?:s)?)$",
    r"^(\d+\.?\s+(?:discussion|analysis))$",
    r"^(\d+\.?\s+(?:conclusion|conclusions|concluding\s+remarks))$",
    r"^(\d+\.?\s+(?:limitation(?:s)?|future\s+work))$",
    r"^(\d+\.?\s+(?:acknowledgment(?:s)?|acknowledgement(?:s)?))$",
    r"^(\d+\.?\s+reference(?:s)?)$",
    r"^(appendix.*)$",
]

# Which sections are relevant for each agent
METHOD_SECTIONS = {
    "abstract", "introduction", "related work", "background",
    "method", "methodology", "model", "approach", "framework",
    "system", "architecture", "proposed",
}
EXPERIMENT_SECTIONS = {
    "abstract", "experiment", "experiments", "experimental setup",
    "experimental results", "experimental evaluation",
    "result", "results", "discussion", "analysis",
}
CRITIC_SECTIONS = {
    "abstract", "introduction", "related work", "conclusion",
    "conclusions", "concluding remarks", "limitation", "limitations",
    "future work", "discussion",
}


@dataclass
class Section:
    title: str
    content: str
    page_start: int
    page_end: int


@dataclass
class ParsedPaper:
    title: str
    full_text: str
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def get_sections_for_agent(self, agent_type: str) -> str:
        """Return concatenated section text relevant to a given agent type."""
        if agent_type == "method":
            relevant = METHOD_SECTIONS
        elif agent_type == "experiment":
            relevant = EXPERIMENT_SECTIONS
        elif agent_type == "critic":
            relevant = CRITIC_SECTIONS
        else:
            return self.full_text

        parts = []
        for sec in self.sections:
            normalized = _normalize_title(sec.title)
            if any(kw in normalized for kw in relevant):
                parts.append(f"## {sec.title}\n{sec.content}")

        # Fall back to full text if no sections matched
        if not parts:
            return self.full_text

        return "\n\n".join(parts)


def _normalize_title(title: str) -> str:
    """Lowercase and strip leading numbering from a section title."""
    title = title.lower().strip()
    title = re.sub(r"^\d+\.?\s*", "", title)
    return title


def _is_section_header(line: str) -> Optional[str]:
    """Return the matched section name if line looks like a section header, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    normalized = stripped.lower()
    for pattern in SECTION_PATTERNS:
        if re.match(pattern, normalized, re.IGNORECASE):
            return stripped
    return None


def parse_pdf(pdf_path: str | Path) -> ParsedPaper:
    """Extract text from a PDF and split it into sections."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))

    # --- Extract metadata ---
    meta = doc.metadata or {}
    paper_title = meta.get("title", "").strip() or pdf_path.stem

    # --- Extract full text page by page ---
    pages_text: List[tuple[int, str]] = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages_text.append((page_num, text))

    doc.close()

    full_text = "\n".join(t for _, t in pages_text)

    # --- Split into sections ---
    sections = _split_into_sections(pages_text)

    # If splitting failed, create a single catch-all section
    if not sections:
        sections = [Section(
            title="Full Paper",
            content=full_text,
            page_start=0,
            page_end=len(pages_text) - 1,
        )]

    return ParsedPaper(
        title=paper_title,
        full_text=full_text,
        sections=sections,
        metadata={k: str(v) for k, v in meta.items() if v},
    )


def _split_into_sections(pages_text: List[tuple[int, str]]) -> List[Section]:
    """Walk through page text and split at detected section headers."""
    sections: List[Section] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    current_page_start = 0
    current_page = 0

    for page_num, page_text in pages_text:
        for line in page_text.splitlines():
            header = _is_section_header(line)
            if header:
                # Save the previous section
                if current_title is not None and current_lines:
                    sections.append(Section(
                        title=current_title,
                        content="\n".join(current_lines).strip(),
                        page_start=current_page_start,
                        page_end=current_page,
                    ))
                current_title = header
                current_lines = []
                current_page_start = page_num
            else:
                if current_title is not None:
                    current_lines.append(line)
        current_page = page_num

    # Save the last section
    if current_title and current_lines:
        sections.append(Section(
            title=current_title,
            content="\n".join(current_lines).strip(),
            page_start=current_page_start,
            page_end=current_page,
        ))

    return sections
