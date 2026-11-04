#!/usr/bin/env python3
"""Rotate PDF pages by quarter turns without modifying their image data.

This module changes each page's PDF ``/Rotate`` attribute, rather than
rendering and re-encoding any page content. Consequently, embedded scan images,
vector drawing commands, and their intrinsic resolution are retained.

Examples
--------
Rotate all pages 90 degrees clockwise, updating the source incrementally::

    python rotate_pdf.py scans.pdf --turns 1

Rotate all pages 90 degrees counterclockwise to a new file::

    python rotate_pdf.py scans.pdf --turns -1 --output upright.pdf

Rotate 180 degrees::

    python rotate_pdf.py scans.pdf --turns 2

Notes
-----
PDF rotation is interpreted by PDF viewers and downstream software, and is not
baked into the scanned pixels. Some software may ignore PDF rotation metadata.
Only exact multiples of 90 degrees are supported. In-place updates append an
incremental revision, keeping original PDF objects untouched; to rewrite to a
new output file requires supplying ``--output``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import fitz


def rotate_pdf(input_file: Path, turns: int, output_file: Path | None = None) -> None:
    """Rotate every PDF page losslessly by the requested quarter turns.

    Parameters
    ----------
    input_file : pathlib.Path
        Existing source PDF file.
    turns : int
        Number of 90-degree clockwise turns. Negative values rotate
        counterclockwise. Rotation is relative to each page's current rotation.
    output_file : pathlib.Path or None, optional
        Destination path. When omitted, save an incremental update in place,
        preserving the source's original PDF data and adding only new objects
        for the changed page metadata. If incremental saving is unavailable,
        raise an error rather than silently rewriting the input file.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the input and output paths are the same or the PDF cannot be opened.
    RuntimeError
        If in-place incremental saving is unsupported for the input PDF.

    Notes
    -----
    No page is rasterized or recompressed. This preserves the image pixel data
    and resolution, although the PDF's metadata and byte representation change.
    """
    source = input_file.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")

    target = output_file.expanduser().resolve() if output_file else None
    if target == source:
        raise ValueError("For in-place rotation, omit --output (incremental save).")
    if target is not None and target.exists():
        raise FileExistsError(f"Output already exists: {target}")

    angle = (int(turns) * 90) % 360
    with fitz.open(source) as doc:
        if not doc.is_pdf:
            raise ValueError("Input file is not a PDF.")
        if doc.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")

        if angle == 0:
            # Doing nothing is the only fully byte-identical no-op.
            if target is not None:
                import shutil
                shutil.copyfile(source, target)
            print(f"No rotation needed ({len(doc)} pages).")
            return

        if target is None and not doc.can_save_incrementally():
            raise RuntimeError(
                "This PDF cannot be incrementally updated in place. "
                "Specify --output NEW.pdf instead."
            )

        for page in doc:
            page.set_rotation((page.rotation + angle) % 360)

        if target is None:
            # This updates the source PDF without decoding/re-encoding content.
            doc.saveIncr()
            destination = source
        else:
            # Save using original streams, without cleaning, deflation, or
            # garbage collection that might rewrite/recompress scan objects.
            doc.save(target, garbage=0, deflate=False)
            destination = target

        print(f"Rotated {len(doc)} pages {angle} degrees clockwise -> {destination}")


def main() -> int:
    """Parse command-line options and perform lossless PDF page rotation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="Source PDF")
    parser.add_argument(
        "--turns", "-t", type=int, required=True,
        help="Number of quarter turns clockwise (e.g., 1, -1, 2).",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="New output PDF. Omit to update the input incrementally in place.",
    )
    args = parser.parse_args()
    try:
        rotate_pdf(args.input, args.turns, args.output)
    except (OSError, ValueError, RuntimeError, fitz.FileDataError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
