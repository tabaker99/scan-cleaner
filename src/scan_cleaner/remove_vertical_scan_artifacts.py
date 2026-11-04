"""Conservatively remove long vertical scanner shadows outside detected text.

The filter identifies vertical shadows from a robust column-wise paper-brightness
profile. It operates only in the outer margins, and, when available, respects
trusted text-line bounds. Text, diagrams, and unknown marks inside the protected
content area are never altered.

This is a standalone module: it has no dependency on the main PDF cleaner and
may be imported with ``from .remove_vertical_scan_artifacts import ...``.
"""

from __future__ import annotations

import cv2
import numpy as np


def remove_vertical_edge_artifacts(
    rgb: np.ndarray,
    text_lines: list[tuple] | None = None,
    edge_fraction: float = 0.18,
    paper_threshold: float = 252.0,
    safety_fraction: float = 0.012,
) -> np.ndarray:
    """Replace persistent vertical edge shadows with white paper.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB ``uint8`` image, typically deskewed and cropped, **before** local
        contrast enhancement.
    text_lines : list of tuple, optional
        Trusted text-line bounds ``(x0, y0, x1, y1, component_count)`` in the
        coordinates of ``rgb``. Columns containing detected text are protected.
        Without lines, the middle 70% of the page is protected instead.
    edge_fraction : float, optional
        Largest fraction of image width to consider from each outer edge.
    paper_threshold : float, optional
        A column's robust paper-brightness value must fall below this level to
        qualify as a dark vertical shadow.
    safety_fraction : float, optional
        Width of an extra gap between the removal region and trusted text.

    Returns
    -------
    numpy.ndarray
        New RGB array with eligible edge shadows replaced by pure white.

    Notes
    -----
    We take the 65th-percentile gray value down each column. This measures
    persistent edge shading but mostly ignores horizontal printed text lines,
    which cover much less than 35% of a typical column's page height. A short
    horizontal copier band is likewise unlikely to trigger detection.

    The profile is Gaussian-smoothed. Each side is examined from the outside
    inward and only contiguous candidate shadow columns near that edge are
    removed. The outermost light pixels are included to remove thin page-edge
    lines; the detector will not extend through a broad light interval.

    Warnings
    --------
    The function is intentionally conservative: a crease inside actual text
    must be fixed with more sophisticated inpainting or manually, not by
    whitening text-bearing columns. It also avoids removing vertical rules
    running through a printed diagram or table.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected H x W x 3 uint8 RGB image')
    height, width = rgb.shape[:2]
    if height < 50 or width < 100:
        return rgb.copy()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Sample rows in the central 90% so that top/bottom copier bars have less
    # influence on the column-wise paper brightness.
    sample = gray[int(.05 * height):int(.95 * height)]
    paper_profile = np.percentile(sample, 65, axis=0).astype(np.float32)
    sigma = max(2.0, width * 0.002)
    smooth = cv2.GaussianBlur(paper_profile[None, :], (0, 0), sigmaX=sigma)[0]

    if text_lines:
        # Only coherent lines are used; scan streaks rejected by text_bbox do
        # not cause us to protect arbitrary artifact columns.
        left_text = min(int(line[0]) for line in text_lines)
        right_text = max(int(line[2]) for line in text_lines)
    else:
        left_text, right_text = int(width * .15), int(width * .85)
    safety = max(3, int(width * safety_fraction))
    allowed_left = max(0, left_text - safety)
    allowed_right = min(width, right_text + safety)
    max_band = max(1, int(width * edge_fraction))

    # Identify persistent dark runs anywhere in the outer margin, including
    # slight inset spine creases with some white paper outside the line.
    result = rgb.copy()
    for lo, hi in ((0, min(max_band, allowed_left)),
                   (max(width - max_band, allowed_right), width)):
        if hi - lo < 3:
            continue
        dark = smooth[lo:hi] < paper_threshold
        if not np.any(dark):
            continue
        # Only erase runs long enough to be real scanner shadows rather than
        # isolated dark glyphs. Expand by the small smoothing radius to catch
        # gray fringes while never crossing into the trusted text region.
        boundaries = np.flatnonzero(np.diff(np.r_[False, dark, False].astype(np.int8)))
        for start, stop in zip(boundaries[::2], boundaries[1::2]):
            if stop - start < max(2, round(width * .001)):
                continue
            a = max(lo, lo + int(start) - int(sigma))
            b = min(hi, lo + int(stop) + int(sigma))
            result[:, a:b] = 255
    return result
