#!/usr/bin/env python3
"""Split correctly oriented, two-page book scans at their spine crease.

The input must show two upright pages side-by-side (landscape spreads). This
program deliberately does not auto-rotate or deskew the source document.

For each spread, a low-resolution grayscale preview is used to locate a
vertical, dark, continuous gutter near the image center. When that is absent,
a wide white corridor is sought and split at its midpoint; otherwise the
geometric midpoint is used. The chosen split is
then mapped back to the PDF page coordinates, and two single-page PDF pages
are created without rasterizing the original PDF content.

The split is a straight vertical cut. This is suitable for most flatbed book
scans but does not dewarp curved text or remove shadows from either page.

Examples
--------
Split an ordinary upright book-scanning PDF::

    python split_book_spreads.py spreads.pdf pages.pdf

Remove 1 mm of shadow from each side of the crease::

    python split_book_spreads.py spreads.pdf pages.pdf --gutter-mm 1

Show estimates without producing an output file::

    python split_book_spreads.py spreads.pdf pages.pdf --dry-run
"""

from __future__ import annotations

import argparse
import json
from deferred_gutter import trace_gutter_shadow
from pathlib import Path

import fitz
import numpy as np


def smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a one-dimensional profile with a Gaussian kernel.

    Parameters
    ----------
    values : numpy.ndarray
        The one-dimensional signal to smooth.
    sigma : float
        Gaussian standard deviation in profile pixels.

    Returns
    -------
    numpy.ndarray
        Smoothed signal, with identical shape to ``values``.
    """
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    return np.convolve(values, weights, mode="same")


def locate_spine(gray: np.ndarray, search_fraction: float = 0.19,
                 min_continuity: float = 0.45) -> tuple[int, float]:
    """Find the strongest vertical gutter in a grayscale book spread.

    The central portion of the image is searched for a dark vertical line.
    Per-column intensity is evaluated across many rows, so ordinary printed
    words contribute less than a continuous spine shadow. The evaluation
    excludes the top and bottom 10 percent, where scanner-edge shadows are
    frequent. To resist a few very dark marks, the darkness contribution of
    each pixel is capped and a continuity measure is also used.

    Parameters
    ----------
    gray : numpy.ndarray
        Two-dimensional uint8 grayscale spread, already correctly oriented.
    search_fraction : float, default=0.19
        Half-width of the search range relative to image width. For example,
        0.19 searches between 31% and 69% of the spread width.
    min_continuity : float, default=0.45
        Minimum fraction of interior page rows showing a gray or dark gutter
        band. A spurious text column can score highly yet fail this test.
        When no candidate meets it, the caller tests for a white corridor.

    Returns
    -------
    split_x : int
        Column index of the estimated gutter center.
    confidence : float
        Contrast score between the selected gutter and the surrounding
        search area, measured in units of the search-area standard deviation.
        This is a diagnostic measure rather than a calibrated probability.

    Raises
    ------
    ValueError
        If the input image is invalid or too small.

    Notes
    -----
    The dark-gutter detector first checks whether a vertically continuous
    dark band exists. ``split_pdf`` subsequently checks for a central white
    corridor when the dark-gutter result is absent or unconvincing. If neither
    is reliable, it uses the geometric midpoint.
    """
    if gray.ndim != 2 or min(gray.shape) < 80:
        raise ValueError("Expected a 2-D grayscale image at least 80 pixels wide/high")

    if not 0.0 < min_continuity <= 1.0:
        raise ValueError("min_continuity must lie in (0, 1]")

    height, width = gray.shape
    interior = gray[int(height * 0.10):int(height * 0.90)].astype(np.float32)
    # Avoid giving isolated solid black spots excessive influence.
    darkness = np.clip(220.0 - interior, 0.0, 140.0)
    mean_ink = darkness.mean(axis=0)
    # A continuous shadow is dark in a large fraction of rows; a heading isn't.
    continuity = (interior < 215).mean(axis=0)
    sigma = max(1.0, width * .002)
    # Require a consistent *vertical band*, not merely a prominent dark mark.
    # Smooth continuity across adjacent columns to suppress isolated strokes.
    band_continuity = smooth_1d(continuity, sigma=sigma)
    score = smooth_1d(mean_ink + 28.0 * continuity, sigma=sigma)

    center = width * .5
    left = max(1, int(center - search_fraction * width))
    right = min(width - 1, int(center + search_fraction * width))
    candidates = score[left:right]
    present = band_continuity[left:right] >= min_continuity
    if not np.any(present):
        # No detectable gutter: the maximum-score column is meaningless.
        return width // 2, 0.0
    eligible = np.where(present, candidates, -np.inf)
    split_x = left + int(np.argmax(eligible))
    confidence = float((score[split_x] - np.median(candidates)) /
                       (np.std(candidates) + 1e-6))
    return split_x, confidence



def locate_white_corridor(gray: np.ndarray, search_fraction: float = 0.19,
                          min_white_fraction: float = 0.965,
                          min_width_fraction: float = 0.025,
                          max_center_offset: float = 0.08) -> tuple[int, float] | None:
    """Find the midpoint of a genuinely blank central vertical corridor.

    This is a fallback for spreads *without a visible dark spine*. A corridor
    must be nearly white over most of the page height, sufficiently wide to
    distinguish it from an incidental inter-letter gap, and close to the
    geometric center. Unlike a darkest-column detector, this identifies the
    **two edges** of a white region and bisects the space between them.

    Returns ``(column, white_coverage)`` or ``None`` if no reliable corridor
    is present. The caller then falls back to the geometric midpoint.
    """
    if gray.ndim != 2 or min(gray.shape) < 80:
        raise ValueError('Expected a 2-D grayscale image at least 80 pixels wide/high')
    h, w = gray.shape
    interior = gray[int(h * .10):int(h * .90)]
    # White-pixel support over interior rows, not merely mean intensity:
    # thin text cannot masquerade as a uniformly blank column.
    white_fraction = (interior >= 245).mean(axis=0).astype(np.float64)
    white_fraction = smooth_1d(white_fraction, sigma=max(1.5, w * .002))
    center = w // 2
    left = max(1, int(center - search_fraction * w))
    right = min(w - 1, int(center + search_fraction * w))
    qualifying = white_fraction[left:right] >= min_white_fraction
    boundaries = np.flatnonzero(np.diff(np.r_[False, qualifying, False].astype(np.int8)))
    min_width = max(6, int(np.ceil(w * min_width_fraction)))
    runs = []
    for a, b in zip(boundaries[::2], boundaries[1::2]):
        a, b = int(left + a), int(left + b)
        midpoint = (a + b) / 2.0
        if (b - a >= min_width and abs(midpoint - center) <= max_center_offset * w
                and a > left + 1 and b < right - 1):
            runs.append((a, b, midpoint))
    if not runs:
        return None
    # Prefer a corridor that includes the physical middle, then the nearest
    # and widest. This avoids distant white page margins being chosen instead.
    a, b, midpoint = min(runs, key=lambda r: (
        not (r[0] <= center <= r[1]),
        abs(r[2] - center),
        -(r[1] - r[0])))
    return int(round(midpoint)), float(np.mean(white_fraction[a:b]))

def render_preview(page: fitz.Page, target_width: int = 1200) -> np.ndarray:
    """Render a PDF page as a compact grayscale array for gutter detection.

    Parameters
    ----------
    page : fitz.Page
        Input PDF page, whose displayed orientation must already be upright.
    target_width : int, default=1200
        Approximate rasterized preview width in pixels.

    Returns
    -------
    numpy.ndarray
        Two-dimensional uint8 grayscale preview.
    """
    zoom = target_width / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY,
                          alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def split_pdf(input_path: Path, output_path: Path, *, search_fraction: float = 0.19,
              gutter_mm: float = 0.0, target_width: int = 1200,
              dry_run: bool = False, min_confidence: float = 0.7,
              min_continuity: float = 0.45,
              save_gutter_geometry: bool = True) -> None:
    """Detect spines and export two single-page PDF pages per input spread.

    Parameters
    ----------
    input_path : pathlib.Path
        PDF containing upright two-page spreads.
    output_path : pathlib.Path
        Destination file. The input is never modified.
    search_fraction : float, default=0.19
        Horizontal search half-width as a fraction of spread width.
    gutter_mm : float, default=0.0
        Additional material removed from *each* side of the detected crease.
        A positive value can remove gutter shadow, but may discard text.
    target_width : int, default=1200
        Width of rendered previews used for detection (not output quality).
    dry_run : bool, default=False
        Print split positions without writing a PDF.
    min_confidence : float, default=0.7
        If the detection is weak, use the page center instead of an uncertain
        location. This threshold is heuristic, not a probability.
    min_continuity : float, default=0.45
        Minimum vertical coverage required before claiming a visible gutter.
        If none is detected, look for a blank central corridor first.

    Notes
    -----
    The output duplicates source PDF pages and adjusts their CropBox rather than image rasterization and
    therefore preserves the resolution of scanned PDF images and any original
    vector text. For a nonzero gutter margin, the exclusion is symmetric.
    """
    if not 0.02 <= search_fraction <= 0.45:
        raise ValueError("--search-fraction must lie between 0.02 and 0.45")
    if gutter_mm < 0:
        raise ValueError("--gutter-mm must be nonnegative")
    source = fitz.open(input_path)
    output = fitz.open()
    geometry_pages = []
    for number, page in enumerate(source, 1):
        if page.rect.width <= page.rect.height:
            raise ValueError(
                f"Spread {number} isn't landscape (width <= height). "
                "Rotate the input PDF before running this script."
            )
        preview = render_preview(page, target_width)
        column, confidence = locate_spine(preview, search_fraction, min_continuity)
        gutter_present = confidence >= min_confidence
        if confidence < min_confidence:
            white_corridor = locate_white_corridor(preview, search_fraction)
            if white_corridor is not None:
                column, white_coverage = white_corridor
                print(f"  No reliable dark gutter on spread {number}; "
                      f"using midpoint of central white corridor "
                      f"(white coverage={white_coverage:.1%})")
            else:
                print(f"  No reliable dark gutter or white corridor on spread {number}; "
                      f"using geometric center")
                column = preview.shape[1] // 2
        # Record shadow boundaries on the original upright spread, before
        # splitting. A missing dark gutter must never imply shadow removal.
        shadows = [None, None]
        if gutter_present and save_gutter_geometry and not dry_run:
            shadows = [trace_gutter_shadow(preview,column,'left'),
                       trace_gutter_shadow(preview,column,'right')]
        position = page.rect.width * column / preview.shape[1]
        margin = gutter_mm * 72 / 25.4
        # Use rectangle coordinates in the *displayed* page orientation.
        # Rotated input scans should be pre-rotated and saved by the caller.
        bounds = [fitz.Rect(0, 0, position - margin, page.rect.height),
                  fitz.Rect(position + margin, 0, page.rect.width, page.rect.height)]
        print(f"Spread {number:>2}: spine at {position/page.rect.width:6.2%} "
              f"of width, score={confidence:.2f}; "
              f"left={bounds[0].width:.1f}pt right={bounds[1].width:.1f}pt")
        if dry_run:
            continue
        for side_no, clip in enumerate(bounds):
            if clip.width <= 0:
                raise ValueError(f"Gutter margin too large on spread {number}")
            # `show_pdf_page` does not preserve source page /Rotate metadata.
            # Instead, duplicate the original PDF page and set its CropBox.
            # CropBox coordinates are in the *unrotated* page coordinate
            # system, so transform the displayed split rectangle back with
            # `derotation_matrix` before setting the crop. This preserves
            # image streams, vector content, and the original rotation.
            native_clip = clip * page.derotation_matrix
            output.insert_pdf(source, from_page=number - 1, to_page=number - 1)
            output[-1].set_cropbox(native_clip)
            # Normalized depth uses *split page width*, not full spread.
            samples = shadows[side_no]
            if samples:
                source_width = column if side_no == 0 else preview.shape[1]-column
                samples = [[yf,round(df * source_width / (column if side_no==0 else preview.shape[1]-column),6)]
                           for yf,df in samples]
            geometry_pages.append({'page':len(output)-1,
                                   'edge':'right' if side_no==0 else 'left',
                                   'samples':samples or []})

    if not dry_run:
        output.save(output_path, garbage=4, deflate=True)
        if save_gutter_geometry:
            path = output_path.with_suffix('.gutter.json')
            path.write_text(json.dumps({'version':1,'source':output_path.name,
                                        'page_count':len(output),
                                        'pages':geometry_pages},indent=2))
            print(f'Saved deferred gutter geometry to {path}')
        print(f"Saved {len(output)} single pages to {output_path}")
    output.close()
    source.close()


def main() -> None:
    """Parse command-line options and run the two-page splitter."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Upright two-page spread PDF")
    parser.add_argument("output", type=Path, help="Output single-page PDF")
    parser.add_argument("--search-fraction", type=float, default=.19,
                        help="Half-width of gutter search region (default: 0.19)")
    parser.add_argument("--gutter-mm", type=float, default=0.0,
                        help="Remove this much near the spine on each side")
    parser.add_argument("--preview-width", type=int, default=1200)
    parser.add_argument("--min-confidence", type=float, default=.7)
    parser.add_argument("--min-continuity", type=float, default=.45,
                        help="Minimum fraction of rows covered by a visible gutter (default: 0.45)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-gutter-geometry", action="store_true",
                        help="Disable deferred gutter-geometry sidecar")
    args = parser.parse_args()
    split_pdf(args.input, args.output, search_fraction=args.search_fraction,
              gutter_mm=args.gutter_mm, target_width=args.preview_width,
              dry_run=args.dry_run, min_confidence=args.min_confidence,
              min_continuity=args.min_continuity,
              save_gutter_geometry=not args.no_gutter_geometry)


if __name__ == "__main__":
    main()
