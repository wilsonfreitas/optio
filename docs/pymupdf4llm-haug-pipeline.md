# Haug PDF extraction with pymupdf4llm

This branch is a fresh extraction path for `The Complete Guide to Option Pricing Formulas - Espen Gaarder Haug.pdf`.

It implements the two-stage "Cenario 3" pipeline:

```text
PDF -> pymupdf4llm OCR markdown -> LLM-reviewed markdown
```

## Setup

```bash
.venv/bin/python -m pip install -r requirements-pymupdf4llm.txt
```

The local machine must also have Tesseract available. The current `.venv` has already been tested with `pymupdf4llm==1.27.2.3`.

## Extract raw markdown

Quick sample:

```bash
.venv/bin/python tools/haug_pymupdf4llm_pipeline.py extract --start-page 1 --end-page 3
```

Full book:

```bash
.venv/bin/python tools/haug_pymupdf4llm_pipeline.py extract
```

Outputs are written under `markdown-pymupdf4llm/haug/`:

- `raw-pages/page-0001.md`, one file per PDF page
- `haug-raw.md`, consolidated raw extraction

The output directory is ignored by git because it is generated reference material.

## Clean with LLM

Set credentials and optionally choose a model:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1"
```

Then run:

```bash
.venv/bin/python tools/haug_pymupdf4llm_pipeline.py clean --batch-pages 4
```

Outputs:

- `clean-batches/pages-0001-0004.md`, resumable cleaned batches
- `haug-clean.md`, consolidated cleaned markdown

The cleaning prompt asks the model to repair OCR errors, formulas, tables, and internal references without inventing missing content. Ambiguous formulas or tables should be kept with `[OCR unclear]`.

## Split into chapters

After `haug-clean.md` has been generated and validated, split it into chapter files:

```bash
.venv/bin/python tools/split_haug_clean_chapters.py
```

Outputs are written under `markdown-pymupdf4llm/haug/chapters/`:

- `00-front-matter.md`
- `01-black-scholes-merton.md` through `14-some-useful-formulas.md`
- `15-bibliography.md`
- `16-index.md`
- `README.md`, a manifest with page ranges

The splitter uses explicit PDF page ranges and validates that every source page is covered exactly once.
