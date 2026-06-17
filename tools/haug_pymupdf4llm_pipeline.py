#!/usr/bin/env python3
"""Extract and LLM-clean Haug's option pricing PDF.

Pipeline:
1. PDF -> raw markdown via pymupdf4llm + OCR
2. raw markdown -> cleaned markdown via an LLM, in resumable page batches
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PDF = "The Complete Guide to Option Pricing Formulas - Espen Gaarder Haug.pdf"
DEFAULT_OUT = "markdown-pymupdf4llm/haug"
PAGE_END_RE = re.compile(r"--- end of page\.page_number=(\d+) ---")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def page_list(start: int | None, end: int | None) -> list[int] | None:
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise SystemExit("--start-page and --end-page must be used together")
    if end < start:
        raise SystemExit("--end-page must be greater than or equal to --start-page")
    return list(range(start - 1, end))


def split_pages(markdown: str) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    offset = 0

    for match in PAGE_END_RE.finditer(markdown):
        page_number = int(match.group(1))
        page_text = markdown[offset : match.end()].strip()
        if page_text:
            pages.append((page_number, page_text + "\n"))
        offset = match.end()

    tail = markdown[offset:].strip()
    if tail:
        inferred = pages[-1][0] + 1 if pages else 1
        pages.append((inferred, tail + "\n"))

    return pages


def page_path(raw_pages_dir: Path, page_number: int) -> Path:
    return raw_pages_dir / f"page-{page_number:04d}.md"


def extract(args: argparse.Namespace) -> None:
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency pymupdf4llm. Install with "
            "`python -m pip install -r requirements-pymupdf4llm.txt`."
        ) from exc

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    out_dir = Path(args.out_dir)
    raw_pages_dir = out_dir / "raw-pages"
    raw_pages_dir.mkdir(parents=True, exist_ok=True)

    pages = page_list(args.start_page, args.end_page)
    markdown = pymupdf4llm.to_markdown(
        str(pdf),
        pages=pages,
        page_separators=True,
        force_ocr=args.force_ocr,
        ocr_dpi=args.ocr_dpi,
        show_progress=True,
    )

    extracted_pages = split_pages(markdown)
    if not extracted_pages:
        raise SystemExit("No page content was extracted.")

    for page_number, page_markdown in extracted_pages:
        page_path(raw_pages_dir, page_number).write_text(page_markdown, encoding="utf-8")

    full_raw = out_dir / "haug-raw.md"
    full_raw.write_text(
        "\n".join(page_markdown.strip() for _, page_markdown in extracted_pages) + "\n",
        encoding="utf-8",
    )

    first_page = extracted_pages[0][0]
    last_page = extracted_pages[-1][0]
    print(f"Wrote {len(extracted_pages)} raw pages ({first_page}-{last_page}) to {raw_pages_dir}")
    print(f"Wrote consolidated raw markdown to {full_raw}")


@dataclass(frozen=True)
class Batch:
    first_page: int
    last_page: int
    text: str

    @property
    def filename(self) -> str:
        return f"pages-{self.first_page:04d}-{self.last_page:04d}.md"


def load_batches(raw_pages_dir: Path, batch_pages: int) -> list[Batch]:
    files = sorted(raw_pages_dir.glob("page-*.md"))
    if not files:
        raise SystemExit(f"No raw page files found in {raw_pages_dir}. Run `extract` first.")

    batches: list[Batch] = []
    for index in range(0, len(files), batch_pages):
        group = files[index : index + batch_pages]
        page_numbers = [int(path.stem.removeprefix("page-")) for path in group]
        text = "\n\n".join(path.read_text(encoding="utf-8").strip() for path in group) + "\n"
        batches.append(Batch(page_numbers[0], page_numbers[-1], text))
    return batches


def cleaning_prompt() -> str:
    return """You are cleaning OCR markdown extracted from a technical finance textbook.

Rules:
- Preserve the original meaning and page order.
- Preserve every page separator exactly in this form: --- end of page.page_number=N ---
- Every input page must produce exactly one matching page separator in the output.
- Fix OCR spelling errors, broken words, duplicated characters, bad line wraps, and malformed headings.
- Repair tables into readable GitHub Flavored Markdown when the source is clearly tabular.
- Repair mathematical formulas into LaTeX where possible.
- Do not invent missing formulas, numbers, references, page text, or citations.
- If a formula or table is too ambiguous to repair, keep the uncertain text and add [OCR unclear].
- Return only cleaned markdown, with no commentary before or after.
"""


def clean_batch(
    client: object,
    model: str,
    batch: Batch,
    temperature: float,
    max_retries: int,
    retry_seconds: float,
) -> str:
    expected_pages = list(range(batch.first_page, batch.last_page + 1))
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": cleaning_prompt()},
                    {
                        "role": "user",
                        "content": (
                            f"Clean this OCR markdown batch from pages "
                            f"{batch.first_page}-{batch.last_page}:\n\n{batch.text}"
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError(f"Empty LLM response for pages {batch.first_page}-{batch.last_page}")
            cleaned = content.strip() + "\n"
            actual_pages = [int(page) for page in PAGE_END_RE.findall(cleaned)]
            if actual_pages != expected_pages:
                raise ValueError(
                    f"Expected page separators {expected_pages}, got {actual_pages}"
                )
            return cleaned
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = status_code == 429 or isinstance(exc, ValueError)
            if not retryable or attempt >= max_retries:
                raise
            wait = retry_seconds * (attempt + 1)
            reason = "rate limited" if status_code == 429 else "invalid page separators"
            print(
                f"{reason.capitalize()} on pages {batch.first_page}-{batch.last_page}; "
                f"retrying in {wait:.1f}s ({attempt + 1}/{max_retries})",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError("unreachable retry loop state")


def clean(args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency openai. Install with "
            "`python -m pip install -r requirements-pymupdf4llm.txt`."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    out_dir = Path(args.out_dir)
    raw_pages_dir = out_dir / "raw-pages"
    clean_batches_dir = out_dir / "clean-batches"
    clean_batches_dir.mkdir(parents=True, exist_ok=True)

    batches = load_batches(raw_pages_dir, args.batch_pages)
    client = OpenAI(api_key=api_key)

    for batch in batches:
        destination = clean_batches_dir / batch.filename
        if destination.exists() and not args.overwrite:
            print(f"Skipping existing {destination}", flush=True)
            continue

        print(f"Cleaning pages {batch.first_page}-{batch.last_page} -> {destination}", flush=True)
        cleaned = clean_batch(
            client,
            args.model,
            batch,
            args.temperature,
            args.max_retries,
            args.retry_seconds,
        )
        destination.write_text(cleaned, encoding="utf-8")
        time.sleep(args.pause_seconds)

    consolidated = out_dir / "haug-clean.md"
    parts = [path.read_text(encoding="utf-8").strip() for path in sorted(clean_batches_dir.glob("pages-*.md"))]
    if parts:
        consolidated.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        print(f"Wrote consolidated cleaned markdown to {consolidated}", flush=True)
    else:
        print("No cleaned batches were available to consolidate.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the source PDF.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Output directory for generated markdown.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract raw markdown with pymupdf4llm.")
    extract_parser.add_argument("--start-page", type=positive_int, help="1-based first PDF page to extract.")
    extract_parser.add_argument("--end-page", type=positive_int, help="1-based last PDF page to extract.")
    extract_parser.add_argument("--ocr-dpi", type=positive_int, default=300, help="OCR DPI.")
    extract_parser.add_argument(
        "--force-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force OCR even if text is detected.",
    )
    extract_parser.set_defaults(func=extract)

    clean_parser = subparsers.add_parser("clean", help="Clean raw markdown batches with an LLM.")
    clean_parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1"))
    clean_parser.add_argument("--batch-pages", type=positive_int, default=4)
    clean_parser.add_argument("--temperature", type=float, default=0)
    clean_parser.add_argument("--pause-seconds", type=float, default=0.5)
    clean_parser.add_argument("--max-retries", type=int, default=8)
    clean_parser.add_argument("--retry-seconds", type=float, default=10.0)
    clean_parser.add_argument("--overwrite", action="store_true")
    clean_parser.set_defaults(func=clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
