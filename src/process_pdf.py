#!/usr/bin/env python3
"""Process scanned two-page spreads, preserving deferred gutter geometry.

Example:
    python src/process_pdf.py 'book_spreads.pdf' cleaned.pdf --turns -1

`--turns 0` assumes that the input already displays upright landscape pages.
Intermediate files and the gutter sidecar are generated under a chosen workdir.
All production dependencies are in src/; scripts/ is reserved for experiments.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

SRC = Path(__file__).resolve().parent


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--turns', type=int, default=0, help='clockwise 90-degree turns to make input spreads upright')
    p.add_argument('--workdir', type=Path, default=None)
    p.add_argument('--processing-dpi', type=int, default=320)
    p.add_argument('--output-dpi', type=int, default=200)
    args=p.parse_args()
    work=args.workdir or (args.output.parent / (args.output.stem+'_work'))
    work.mkdir(parents=True,exist_ok=True)
    upright = work/'upright.pdf'
    split = work/'split_with_gutter.pdf'
    if args.turns:
        run(SRC/'rotate_pdf.py',args.input,'--turns',args.turns,'--output',upright)
    else:
        upright = args.input
    run(SRC/'split_book_spreads.py',upright,split)
    sidecar=split.with_suffix('.gutter.json')
    if not sidecar.is_file():
        raise FileNotFoundError(f'Gutter geometry missing after splitting: {sidecar}')
    run(SRC/'clean_scanned_pdf.py',split,args.output,
        '--dpi',args.processing_dpi,'--output-dpi',args.output_dpi,
        '--remove-slanted-edge-artifacts','--require-gutter-geometry',
        '--gutter-geometry',sidecar)

if __name__=='__main__':
    main()
