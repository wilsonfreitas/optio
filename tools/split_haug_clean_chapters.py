#!/usr/bin/env python3
"""Split the cleaned Haug markdown into front matter, chapters, and back matter."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = "markdown-pymupdf4llm/haug/haug-clean.md"
DEFAULT_OUT_DIR = "markdown-pymupdf4llm/haug/chapters"
PAGE_END_RE = re.compile(r"--- end of page\.page_number=(\d+) ---")


@dataclass(frozen=True)
class Section:
    filename: str
    title: str
    start_page: int
    end_page: int
    kind: str = "chapter"


SECTIONS = [
    Section("00-front-matter.md", "Front Matter", 1, 33, "front-matter"),
    Section("01-black-scholes-merton.md", "Chapter 1: Black-Scholes-Merton", 34, 52),
    Section("02-black-scholes-merton-greeks.md", "Chapter 2: Black-Scholes-Merton Greeks", 53, 136),
    Section("03-analytical-formulas-for-american-options.md", "Chapter 3: Analytical Formulas for American Options", 137, 150),
    Section("04-exotic-options-single-asset.md", "Chapter 4: Exotic Options - Single Asset", 151, 242),
    Section("05-exotic-options-on-two-assets.md", "Chapter 5: Exotic Options on Two Assets", 243, 272),
    Section("06-black-scholes-merton-adjustments-and-alternatives.md", "Chapter 6: Black-Scholes-Merton Adjustments and Alternatives", 273, 318),
    Section("07-trees-and-finite-difference-methods.md", "Chapter 7: Trees and Finite Difference Methods", 319, 383),
    Section("08-monte-carlo-simulation.md", "Chapter 8: Monte Carlo Simulation", 384, 405),
    Section("09-options-on-stocks-that-pay-discrete-dividends.md", "Chapter 9: Options on Stocks That Pay Discrete Dividends", 406, 434),
    Section("10-commodity-and-energy-options.md", "Chapter 10: Commodity and Energy Options", 435, 449),
    Section("11-interest-rate-derivatives.md", "Chapter 11: Interest Rate Derivatives", 450, 480),
    Section("12-volatility-and-correlation.md", "Chapter 12: Volatility and Correlation", 481, 500),
    Section("13-distributions.md", "Chapter 13: Distributions", 501, 522),
    Section("14-some-useful-formulas.md", "Chapter 14: Some Useful Formulas", 523, 534),
    Section("15-bibliography.md", "Bibliography", 535, 554, "back-matter"),
    Section("16-index.md", "Index", 555, 572, "back-matter"),
]


def split_pages(markdown: str) -> dict[int, str]:
    pages: dict[int, str] = {}
    offset = 0

    for match in PAGE_END_RE.finditer(markdown):
        page_number = int(match.group(1))
        page_text = markdown[offset : match.end()].strip() + "\n"
        if page_number in pages:
            raise ValueError(f"Duplicate page marker: {page_number}")
        pages[page_number] = page_text
        offset = match.end()

    tail = markdown[offset:].strip()
    if tail:
        raise ValueError("Unexpected content after final page marker")

    return pages


def validate_coverage(sections: list[Section], pages: dict[int, str]) -> None:
    expected_pages = set(pages)
    covered: list[int] = []

    for section in sections:
        if section.end_page < section.start_page:
            raise ValueError(f"Invalid range for {section.filename}")
        covered.extend(range(section.start_page, section.end_page + 1))

    covered_set = set(covered)
    duplicate_pages = sorted(page for page in covered_set if covered.count(page) > 1)
    missing_pages = sorted(expected_pages - covered_set)
    extra_pages = sorted(covered_set - expected_pages)

    if duplicate_pages:
        raise ValueError(f"Section ranges overlap on pages: {duplicate_pages}")
    if missing_pages:
        raise ValueError(f"Section ranges do not cover pages: {missing_pages}")
    if extra_pages:
        raise ValueError(f"Section ranges reference missing source pages: {extra_pages}")


def section_markdown(section: Section, pages: dict[int, str]) -> str:
    body = "\n".join(
        pages[page].strip() for page in range(section.start_page, section.end_page + 1)
    )
    return (
        f"# {section.title}\n\n"
        f"<!-- kind: {section.kind}; pdf_pages: {section.start_page}-{section.end_page} -->\n\n"
        f"{body}\n"
    )


def write_manifest(out_dir: Path, sections: list[Section]) -> None:
    rows = [
        "# Haug Clean Markdown Chapter Manifest",
        "",
        "| File | Title | Kind | PDF pages |",
        "|---|---|---|---|",
    ]
    for section in sections:
        rows.append(
            f"| `{section.filename}` | {section.title} | {section.kind} | "
            f"{section.start_page}-{section.end_page} |"
        )
    rows.append("")
    (out_dir / "README.md").write_text("\n".join(rows), encoding="utf-8")


def split(input_path: Path, out_dir: Path) -> None:
    markdown = input_path.read_text(encoding="utf-8")
    pages = split_pages(markdown)
    validate_coverage(SECTIONS, pages)

    out_dir.mkdir(parents=True, exist_ok=True)
    for section in SECTIONS:
        destination = out_dir / section.filename
        destination.write_text(section_markdown(section, pages), encoding="utf-8")
        print(f"Wrote {destination} ({section.start_page}-{section.end_page})")

    write_manifest(out_dir, SECTIONS)
    print(f"Wrote {out_dir / 'README.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Clean consolidated markdown file.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for chapter markdown files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    split(Path(args.input), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
