"""CLI entry point for the multi-agent paper reader."""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Paper Reader — generates structured reading notes from a PDF."
    )
    parser.add_argument("pdf", type=str, help="Path to the academic paper PDF")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save the JSON output (default: print to stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Lazy imports so startup is fast for --help
    from core.graph import run_pipeline
    from core.pdf_parser import parse_pdf

    print(f"Parsing PDF: {pdf_path.name} ...", file=sys.stderr)
    parsed = parse_pdf(pdf_path)
    print(f"  Title   : {parsed.title}", file=sys.stderr)
    print(f"  Sections: {len(parsed.sections)}", file=sys.stderr)

    print("Running multi-agent pipeline (3 agents in parallel) ...", file=sys.stderr)
    summary = run_pipeline(parsed)

    indent = 2 if args.pretty else None
    output_json = summary.model_dump_json(indent=indent)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"Output saved to: {out_path}", file=sys.stderr)
    else:
        print(output_json)

    # Also print a human-readable summary to stderr
    print("\n" + "=" * 60, file=sys.stderr)
    print("PAPER READING NOTES", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Title   : {parsed.title}", file=sys.stderr)
    print(f"Summary : {summary.one_sentence_summary}", file=sys.stderr)
    print("\nCore Contributions:", file=sys.stderr)
    for i, c in enumerate(summary.core_contributions, 1):
        print(f"  {i}. {c}", file=sys.stderr)
    print(f"\nNovelty : {summary.method_highlights}", file=sys.stderr)
    print(f"\nResults : {summary.experiment_highlights}", file=sys.stderr)
    print(f"\nLimits  : {summary.limitations_and_future_work}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
