# Extract VBA Macros from Excel Spreadsheets

## Problem

The `option-pricing/` directory contains 33 `.xls` spreadsheets (OLE2 format) organized across 13 chapter folders. These contain VBA macros implementing option pricing models. We need a way to extract the VBA source code into standalone files for readability, version control, and reference.

## Solution

A standalone Python script `extract_vba.py` in the project root that:

1. Recursively walks `option-pricing/` and finds all `.xls` files
2. Uses `oletools.olevba.VBA_Parser` to extract VBA modules from each file
3. Writes each module as `<SpreadsheetName>_<ModuleName>.vb` in the same directory as the source spreadsheet
4. Skips files with no VBA content
5. Prints a summary of files processed, modules extracted, and any errors

## Naming Convention

All extracted files use the `.vb` extension, prefixed with the spreadsheet name to avoid conflicts:

```
option-pricing/Chapter1and2/PlainVanilla.xls
  → option-pricing/Chapter1and2/PlainVanilla_Module1.vb
  → option-pricing/Chapter1and2/PlainVanilla_BSFormulas.vb
```

## Dependencies

- `oletools` — installed via `pip install oletools` into the existing `.venv/`

## Error Handling

- If a file can't be parsed or has no VBA, log a warning and continue to the next file
- Print a final summary with counts and any errors encountered
