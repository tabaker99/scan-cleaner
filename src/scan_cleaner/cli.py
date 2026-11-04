"""Command-line entry point for scan-cleaner."""
from __future__ import annotations
import argparse
from .pipeline import process_pdf


def main() -> None:
    p = argparse.ArgumentParser(prog='scan-cleaner', description='Clean scanned two-page book spreads.')
    p.add_argument('input', help='Input PDF of scanned spreads')
    p.add_argument('output', help='Output cleaned half-letter PDF')
    p.add_argument('--turns', type=int, default=0, help='Clockwise quarter turns for upright spreads (default 0)')
    p.add_argument('--workdir', default=None, help='Intermediate PDF and gutter-geometry directory')
    p.add_argument('--processing-dpi', type=int, default=320)
    p.add_argument('--output-dpi', type=int, default=200)
    args = p.parse_args()
    process_pdf(args.input, args.output, turns=args.turns, workdir=args.workdir,
                processing_dpi=args.processing_dpi, output_dpi=args.output_dpi)


if __name__ == '__main__':
    main()
