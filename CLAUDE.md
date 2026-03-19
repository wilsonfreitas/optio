# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project has two parts:

### 1. Reference Material Extraction

Extracting and structuring content from two option pricing textbooks:

- **Haug** — "The Complete Guide to Option Pricing Formulas" (Espen Gaarder Haug, 2nd ed., 2006)
- **Rouah** — "Option Pricing Models and Volatility Using Excel-VBA" (Fabrice D. Rouah)

The project converts scanned/digital PDFs into chapter-wise markdown and collects the accompanying Excel spreadsheets (`.xls`) containing VBA-based pricing models.

### 2. Python Option Pricing Library — `optio`

The main objective is to build **optio**, a comprehensive Python option pricing library — intended to be the "de facto" option pricing library. It implements all pricing functions documented in the Haug book.

**GitHub:** https://github.com/wilsonfreitas/optio

**Development approach:**
- Built chapter by chapter, spreadsheet by spreadsheet
- VBA code from `option-pricing/` (extracted `.vb` files) serves as the starting point for Python implementations
- Markdown files from `markdown/` provide documentation, formulas, and references
- Each pricing function must include clear documentation with references from the book
- The markdown files may contain OCR errors (especially formulas) — preprocess chapter by chapter before use, do not load all chapters at once

**Code philosophy:**
- Functional approach — the library follows functional programming patterns
- Pure functions, immutability where practical, composition over inheritance
- Designed with a future API layer in mind
- numpy-vectorized: all inputs coerced to ndarray internally via `broadcast_args()`, scalar inputs yield 0-d arrays

**API design:**
- Flat API with model prefixes: `bs_price`, `bs_delta`, `black_price`, `gk_price`, etc.
- Call/put via string flag: `"call"` / `"put"` (supports vectorized arrays of flags)
- Cost-of-carry parameter `b` kept explicit per Haug's formulation
- Configurable differentiation: `settings.diff_method` = `"analytical"` | `"finite_diff"` | `"ad"` (future)
- Context manager for temporary overrides: `with settings.override(diff_method="finite_diff")`

**Initial release scope:** distributions (`nd`, `cnd`, `cndev`, `cbnd`) + Chapters 1–2 (Generalized Black-Scholes, Black 76, Garman-Kohlhagen)

**Design & plan documents:**
- Spec: `docs/superpowers/specs/2026-03-18-optio-library-design.md`
- Plan: `docs/superpowers/plans/2026-03-18-optio-initial-release.md` (12 tasks, 5 chunks, TDD)

## Repository Structure

- `optio/` — Python option pricing library package (published to GitHub, target PyPI)
- `docs/superpowers/specs/` — Design specifications
- `docs/superpowers/plans/` — Implementation plans
- `extract_to_markdown.py` — OCR-based extraction of scanned Haug PDF → `markdown/` (uses PyMuPDF + Tesseract at 300 DPI)
- `extract_rouah.py` — Text-based extraction of Rouah PDF → `markdown-rouah/` (uses pypdf, no OCR needed)
- `extract_vba.py` — Extracts VBA macros from `.xls` spreadsheets → `.vb` files alongside source spreadsheets
- `markdown/` — 16 chapter markdown files from Haug (chapters 00–15)
- `markdown-rouah/` — 16 chapter markdown files from Rouah (chapters 00–15)
- `option-pricing/` — 33 Excel `.xls` spreadsheets organized by chapter (Chapter1and2 through Chapter14), containing VBA macros for option pricing models, plus 332 extracted `.vb` files

**Note:** Only `optio/`, `docs/`, `CLAUDE.md`, and `.gitignore` are tracked in git. Reference material (PDFs, spreadsheets, VBA files, markdown extractions, extraction scripts) is excluded via `.gitignore`.

## Running the Extraction Scripts

All scripts are standalone and expect PDFs in the project root:

```bash
# Haug (OCR-based, slow — requires tesseract system package)
python extract_to_markdown.py

# Rouah (text extraction, fast)
python extract_rouah.py

# VBA macro extraction from spreadsheets
python extract_vba.py
```

The Python virtual environment is at `.venv/` with Python 3.12. Key dependencies: `pymupdf`, `pytesseract`, `pypdf`, `Pillow`, `torch`, `transformers`, `oletools`.

## Key Design Patterns

- Both extraction scripts use a `CHAPTERS` list mapping chapter metadata (filename, title, book page range) to PDF page ranges via an offset
- Haug extraction requires OCR because the PDF is scanned; Rouah uses native text extraction
- Output markdown includes page separators (`---`) with page numbers for cross-referencing back to the source PDFs
- Library development proceeds chapter by chapter — never load all chapters at once
