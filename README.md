# Scan Cleaner

Clean scanned two-page book spreads into **5.5 × 8.5-inch** pages. Default processing is at 320 DPI, with final output downsampled to 200 DPI.

## Install

```bash
python -m pip install .
# editable development install:
python -m pip install -e .
```

## Run

```bash
scan-cleaner input.pdf output.pdf --turns 0
# or
python -m scan_cleaner input.pdf output.pdf --turns 0
```

Use `--turns -1` or `--turns 1` when scans need rotation. The program keeps intermediates and the required `split_with_gutter.gutter.json` in `<output_stem>_work/` by default. `--workdir` overrides that location.

## Layout

- `src/scan_cleaner/`: installed production library, pipeline and CLI.
- `experiments/`: experimental investigations; never required for production.
- `tests/`: automated tests.
- No `scripts/` folder: all previous experimental scripts live in `experiments/`.

The main pipeline is importable:

```python
from scan_cleaner import process_pdf
process_pdf('input.pdf', 'output.pdf', turns=0)
```

The legacy module command-line tools remain available as `python -m scan_cleaner.rotate_pdf`, `python -m scan_cleaner.split_book_spreads`, and `python -m scan_cleaner.clean_scanned_pdf`.
