"""Production orchestration of rotated spreads, gutter detection and cleanup.

The gutter JSON is mandatory and automatically passed to the cleaner.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def _run_module(module: str, *args: object) -> None:
    subprocess.run([sys.executable, '-m', 'scan_cleaner.' + module,
                    *[str(a) for a in args]], check=True)


def process_pdf(input_path: str | Path, output_path: str | Path,
                *, turns: int = 0, workdir: str | Path | None = None,
                processing_dpi: int = 320, output_dpi: int = 200) -> Path:
    """Run all production steps, preserving the gutter-coordinate sidecar."""
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError('Input and output PDFs must be different')
    if turns not in (-3,-2,-1,0,1,2,3):
        raise ValueError('turns must be between -3 and 3')
    if not 1 <= output_dpi <= processing_dpi:
        raise ValueError('Require 1 <= output_dpi <= processing_dpi')
    output.parent.mkdir(parents=True,exist_ok=True)
    work = Path(workdir).expanduser().resolve() if workdir is not None else output.parent / (output.stem + '_work')
    work.mkdir(parents=True,exist_ok=True)
    if turns:
        upright = work / 'upright.pdf'
        _run_module('rotate_pdf', source, '--turns', turns, '--output', upright)
    else:
        upright = source
    split = work / 'split_with_gutter.pdf'
    _run_module('split_book_spreads', upright, split)
    sidecar = split.with_suffix('.gutter.json')
    if not sidecar.is_file():
        raise FileNotFoundError(f'Gutter geometry missing after splitting: {sidecar}')
    _run_module('clean_scanned_pdf', split, output, '--dpi', processing_dpi,
                '--output-dpi', output_dpi, '--remove-slanted-edge-artifacts',
                '--require-gutter-geometry', '--gutter-geometry', sidecar)
    return output
