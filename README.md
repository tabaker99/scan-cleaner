# Scanned PDF processing

All production code is in `src/`. `scripts/` is reserved for optional, experimental development and is **not required** for processing.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

Scanned, landscape book spreads already upright:

```bash
python src/process_pdf.py input.pdf output.pdf --turns 0
```

For scans that need a quarter-turn, use `--turns 1` or `--turns -1`.

Default processing is 320 DPI with final 200 DPI half-letter pages. The workflow creates intermediate PDFs and a `.gutter.json` sidecar in `<output_stem>_work/`, then **requires** that geometry when cleaning. The sidecar is not optional in the full workflow, even if some spreads have no detectable gutter shadow.

Optional: `--workdir /path/to/work`, `--processing-dpi 320`, `--output-dpi 200`.

### Individual production stages

- `src/rotate_pdf.py`: lossless orientation adjustment
- `src/split_book_spreads.py`: split spreads and save deferred gutter coordinates
- `src/clean_scanned_pdf.py`: deskew, text detection, scale/alignment, masking, contrast enhancement, downsampling, half-letter PDF
- `src/deferred_gutter.py`: text-protected deferred shadow cleanup

Individual modules may be run directly, but `src/process_pdf.py` is preferred because it preserves gutter geometry between stages. The pipeline always splits spreads; it is not a single-page PDF mode.
