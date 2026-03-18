# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project has two parts:

### 1. Reference Material Extraction

Extracting and structuring content from two option pricing textbooks:

- **Haug** — "The Complete Guide to Option Pricing Formulas" (Espen Gaarder Haug, 2nd ed., 2006)
- **Rouah** — "Option Pricing Models and Volatility Using Excel-VBA" (Fabrice D. Rouah)

The project converts scanned/digital PDFs into chapter-wise markdown and collects the accompanying Excel spreadsheets (`.xls`) containing VBA-based pricing models.

### 2. Python Option Pricing Library

The main objective is to build a comprehensive Python option pricing library — intended to be the "de facto" option pricing library. It implements all pricing functions documented in the Haug book.

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

## Repository Structure

- `extract_to_markdown.py` — OCR-based extraction of scanned Haug PDF → `markdown/` (uses PyMuPDF + Tesseract at 300 DPI)
- `extract_rouah.py` — Text-based extraction of Rouah PDF → `markdown-rouah/` (uses pypdf, no OCR needed)
- `extract_vba.py` — Extracts VBA macros from `.xls` spreadsheets → `.vb` files alongside source spreadsheets
- `markdown/` — 16 chapter markdown files from Haug (chapters 00–15)
- `markdown-rouah/` — 16 chapter markdown files from Rouah (chapters 00–15)
- `option-pricing/` — 33 Excel `.xls` spreadsheets organized by chapter (Chapter1and2 through Chapter14), containing VBA macros for option pricing models, plus 332 extracted `.vb` files

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
