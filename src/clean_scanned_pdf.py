#!/usr/bin/env python3
"""Straighten, crop and locally normalize scanned PDF pages.

Requires: pip install pymupdf opencv-python numpy scipy
Example: python clean_scanned_pdf.py input.pdf output.pdf --dpi 320
All output pages are 5.5 x 8.5 inches. Pages with similar central-region
character sizes share a document-wide crop frame to avoid artifact-induced
zoom differences. Unusual layouts are handled conservatively.

Pure image processing: no OCR, no cloud service. Output is an image-only PDF.
"""
# Standard-library modules: CLI parsing, PDF/image geometry, diagnostics, and paths.
import argparse
import math
import sys
from pathlib import Path

# OpenCV supplies raster transformations and connected-component statistics.
# PyMuPDF (fitz) renders input PDF pages and assembles the output document.
# NumPy represents the image buffers; SciPy smooths noisy projection profiles.
import cv2
import fitz
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter
from image_enhancement import (normalize_local, enhance_grayscale,
                               text_contrast_mask, encode_grayscale_jpeg, grid_cell_pixels)


def foreground(gray):
    """Estimate foreground ink using a local illumination baseline.

    A morphological closing operation estimates the slowly varying paper
    background. Pixels sufficiently darker than that estimate are treated as
    ink, allowing shadowed paper to be ignored during geometry analysis.

    Parameters
    ----------
    gray : numpy.ndarray
        Two-dimensional uint8 grayscale scan.

    Returns
    -------
    numpy.ndarray
        Binary uint8 mask with 1 for potential ink and 0 for background.

    Notes
    -----
    The threshold is deliberately conservative. The output is not intended
    as an OCR-quality binarization; it supplies coarse page geometry.
    """
    # A downscaled illumination estimate ensures that shadows aren't seen as ink.
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    contrast = bg.astype(np.int16) - gray.astype(np.int16)
    return (contrast > 28).astype(np.uint8)


def projection_score(mask, angle):
    """Score how strongly page ink aligns with horizontal text lines.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary 2D image of candidate text pixels.
    angle : float
        Counterclockwise rotation angle, in degrees.

    Returns
    -------
    float
        Mean squared first difference of the smoothed horizontal ink
        projection. Larger values indicate more sharply aligned text rows.

    Notes
    -----
    Only the central page region contributes, reducing interference from
    page edges and adjacent-page fragments. The score is a heuristic; it is
    not a calibrated probability of correct orientation.
    """
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Ignore outlying edges (page borders, copier bands).
    central = rotated[int(.08*h):int(.92*h), int(.12*w):int(.88*w)]
    profile = central.sum(axis=1).astype(np.float32)
    profile = gaussian_filter1d(profile, 1.1)
    return float(np.mean(np.diff(profile) ** 2))


def estimate_skew(gray, max_skew=8.0):
    """Estimate one global page rotation by coarse-to-fine projection search.

    Parameters
    ----------
    gray : numpy.ndarray
        Two-dimensional grayscale page image.
    max_skew : float, optional
        Maximum absolute angle to search, in degrees. Default is 8.

    Returns
    -------
    angle : float
        Estimated deskew rotation in degrees, following OpenCV's rotation
        convention.
    confidence : float
        Ratio of the best fine-search score to the median coarse-search
        score. This is a heuristic strength indicator, not a probability.

    Notes
    -----
    A single global rotation cannot correct curved or nonparallel baselines.
    When too little candidate ink is present, return ``(0.0, 0.0)``.
    """
    h, w = gray.shape
    scale = min(1., 900 / max(h, w))
    reduced = cv2.resize(gray, (max(1, int(w*scale)), max(1, int(h*scale))))
    mask = foreground(reduced)
    # Strip long solid borders; text is usually in small separate components.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, np.uint8)
    for k in range(1, n):
        x, y, cw, ch, area = stats[k]
        if 2 <= area <= mask.size * .006 and cw < mask.shape[1] * .5 and ch < mask.shape[0] * .20:
            keep[k] = 1
    mask = keep[labels]
    if np.count_nonzero(mask) < 150:
        return 0., 0.
    coarse = np.arange(-max_skew, max_skew + .001, .5)
    scores = np.array([projection_score(mask, float(a)) for a in coarse])
    best = float(coarse[np.argmax(scores)])
    fine = np.arange(max(-max_skew, best-.6), min(max_skew, best+.6)+.001, .05)
    fs = np.array([projection_score(mask, float(a)) for a in fine])
    selected = float(fine[np.argmax(fs)])
    confidence = float(max(fs) / max(np.median(scores), 1e-6))
    return selected, confidence


def rotate_bound(image, angle, bg=255):
    """Rotate an image without discarding its corners.

    Parameters
    ----------
    image : numpy.ndarray
        Source grayscale or RGB image.
    angle : float
        Counterclockwise rotation angle in degrees.
    bg : int, optional
        Fill value for newly exposed pixels. Default is white (255).

    Returns
    -------
    numpy.ndarray
        Deskewed image on an expanded canvas with white exterior pixels.

    Notes
    -----
    Cubic resampling preserves antialiased glyph strokes but can introduce
    small interpolation halos at hard borders.
    """
    h, w = image.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = math.ceil(h*sin+w*cos), math.ceil(h*cos+w*sin)
    M[0, 2] += nw/2 - center[0]
    M[1, 2] += nh/2 - center[1]
    return cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(bg, bg, bg))


def text_bbox(img, margin=24, return_lines=False, detector="legacy", spatial_filter="height-neighbors"):
    """Find a conservative text-derived crop rectangle without OCR.

    Parameters
    ----------
    img : numpy.ndarray
        Deskewed RGB page image.
    margin : int, optional
        Extra crop margin in source-image pixels, default 24.
    return_lines : bool, optional
        When True, also return retained text-line proposals, default False.

    Returns
    -------
    rect : tuple of int
        Bounding rectangle ``(x0, y0, x1, y1)`` in source pixels with the
        upper-right endpoints exclusive, constrained to image dimensions.
    lines : list of tuple, optional
        Returned only when ``return_lines=True``. Each entry is
        ``(x0, y0, x1, y1, component_count)`` in source pixels.

    Notes
    -----
    Candidate glyphs are joined horizontally to form line proposals. A
    consensus horizontal column and neighboring-line support discard many
    scanner streaks. Isolated page-number glyphs are intentionally *not*
    rescued: artifacts often resemble page numbers. For ambiguous or sparse
    input, this function can return a full-page fallback rectangle.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = min(1., 1100 / max(h, w))
    W, H = max(1, round(w * scale)), max(1, round(h * scale))
    g = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    from text_detection import detect_ink
    fg = detect_ink(g, method=detector)
    # Optional experimentally validated ink-pixel filtering. Cropping defaults
    # to 'off' to avoid changing geometry before multi-page regression checks.
    if spatial_filter != 'off':
        from text_filtering import neighbor_consistency, baseline_projection, union_filter, tuned_baseline, tuned_horizontal_neighbors, height_based_horizontal_neighbors, height_neighbors_with_wide_glyph_rescue
        functions = {'neighbors': neighbor_consistency,
                     'baseline': baseline_projection, 'combined': union_filter,
                     'tuned-baseline': tuned_baseline, 'tuned-neighbors': tuned_horizontal_neighbors, 'height-neighbors': height_based_horizontal_neighbors, 'height-neighbors-rescue': height_neighbors_with_wide_glyph_rescue}
        fg = functions[spatial_filter](g, fg)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)

    # Use character-like connected components; long borders and copier bars
    # cannot enter the text-line detector even if they are close to text.
    chars = np.zeros((H, W), np.uint8)
    for k in range(1, n):
        x, y, cw, ch, area = map(int, stats[k])
        if (2 <= cw <= .075 * W and 2 <= ch <= .035 * H
                and 3 <= area <= .0015 * W * H
                and area >= .065 * cw * ch):
            chars[y:y+ch, x:x+cw][labels[y:y+ch, x:x+cw] == k] = 1

    # Join letters within lines, but not across vertically separated lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                (max(8, round(.022 * W)), max(1, round(.002 * H))))
    joined = cv2.morphologyEx(chars, cv2.MORPH_CLOSE, kernel)
    count, line_labels, line_stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    candidates = []
    for k in range(1, count):
        x, y, cw, ch, area = map(int, line_stats[k])
        if not (.055*W <= cw <= .94*W and 2 <= ch <= .042*H):
            continue
        # A real line should consist of multiple separate characters.
        roi = (labels[y:y+ch, x:x+cw])
        ids = np.unique(roi[chars[y:y+ch, x:x+cw] > 0])
        if len(ids) < 5:
            continue
        ink = int(chars[y:y+ch, x:x+cw].sum())
        if ink < 12 or ink / max(1, cw*ch) > .82:
            continue
        candidates.append((x,y,x+cw,y+ch,len(ids)))
    # Scanner streaks sometimes fragment into dozens of character-sized bits.
    # Such a "line" has far more disconnected fragments and greater vertical
    # irregularity than the genuine text lines below it. Reject only extreme
    # cases near the page edge; ordinary short headings remain eligible.
    if len(candidates) >= 12:
        typical_count = float(np.median([c[4] for c in candidates]))
        typical_height = float(np.median([c[3]-c[1] for c in candidates]))
        candidates = [c for c in candidates if not (
            c[1] < .12*H and c[4] > max(40.,1.9*typical_count)
            and (c[3]-c[1]) > 1.15*typical_height)]
    if not candidates:
        rect = (0, 0, w, h)
        return (rect, []) if return_lines else rect

    # Choose the main text column using horizontal overlap consensus. This
    # rejects snippets of the neighboring book page along the gutter.
    coverage = np.zeros(W, np.float32)
    for x0,y0,x1,y1,nc in candidates:
        coverage[x0:x1] += min(2., nc / 12.)
    coverage = gaussian_filter1d(coverage, sigma=max(2., .013 * W))
    dominant = coverage >= max(1.5, coverage.max() * .36)
    runs = np.flatnonzero(np.diff(np.r_[False,dominant,False].astype(np.int8)))
    if len(runs) >= 2:
        pairs = [(int(a),int(b)) for a,b in zip(runs[::2],runs[1::2])]
        core_l, core_r = max(pairs, key=lambda ab: (ab[1]-ab[0])*float(coverage[ab[0]:ab[1]].mean()))
    else:
        core_l, core_r = 0, W
    retained = [c for c in candidates
                if max(0, min(c[2],core_r)-max(c[0],core_l)) >= .35*(c[2]-c[0])]
    if len(retained) < 2:
        retained = candidates

    # Remove isolated upper/lower "lines" that are far from a dense main body.
    # A noisy copier band often breaks into character-sized fragments, but does
    # not produce neighboring baselines. Preserve sparse title pages (few lines).
    if len(retained) >= 8:
        ordered = sorted(retained, key=lambda c: (c[1]+c[3])/2)
        gaps = [ordered[j+1][1]-ordered[j][3] for j in range(len(ordered)-1)]
        # At least 5 lines support the main body beyond an isolated top band.
        if gaps and gaps[0] > .075*H and len(ordered[1:]) >= 5:
            retained = ordered[1:]
        # Same safeguard for isolated marks at the bottom.
        ordered = sorted(retained, key=lambda c: (c[1]+c[3])/2)
        gaps = [ordered[j+1][1]-ordered[j][3] for j in range(len(ordered)-1)]
        if gaps and gaps[-1] > .075*H and len(ordered[:-1]) >= 5:
            retained = ordered[:-1]

    # The top/bottom come ONLY from character clusters belonging to lines;
    # scanner streaks contribute nothing to these bounds.
    x0 = min(c[0] for c in retained)
    y0 = min(c[1] for c in retained)
    x1 = max(c[2] for c in retained)
    y1 = max(c[3] for c in retained)
    # Do not recover isolated page-number glyphs: noisy copier edges often
    # masquerade as digits. Only coherent detected text lines set crop bounds.
    lines = [(int(a/scale), int(b/scale), int(c/scale), int(d/scale), num)
             for a,b,c,d,num in retained]
    if x1-x0 < .10*W or y1-y0 < .035*H:
        rect = (0,0,w,h)
    else:
        rect = (max(0, int(x0/scale-margin)), max(0, int(y0/scale-margin)),
                min(w, int(x1/scale+margin)), min(h, int(y1/scale+margin)))
    return (rect, lines) if return_lines else rect



def remove_copier_bands(rgb):
    """Suppress wide, densely dark horizontal copier bars.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB image to modify.

    Returns
    -------
    numpy.ndarray
        The same array, possibly modified **in place** by whitening the
        rows of strongly detected copier bands.

    Warnings
    --------
    This operation is distinct from the removed blanket whitening outside
    the text-height range. It still replaces rows, but only when they meet
    stringent dark-coverage and thickness tests. A legitimate very dark,
    full-width image could trigger the heuristic.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    dark_fraction = (g < 115).mean(axis=1)
    suspicious = dark_fraction > .32
    edges = np.flatnonzero(np.diff(np.r_[False, suspicious, False].astype(np.int8)))
    for y0, y1 in zip(edges[::2], edges[1::2]):
        if y1-y0 >= max(8, round(.008*h)):
            # Bands often contain holes; remove entire saturated rows.
            rgb[max(0, y0-2):min(h, y1+2)] = 255
    return rgb


def central_text_scale(rgb):
    """Measure representative glyph height from the page's central 20%.

    Parameters
    ----------
    rgb : numpy.ndarray
        Deskewed RGB page image, before cropping.

    Returns
    -------
    float or None
        Robust estimated glyph height in original page pixels, or None
        when fewer than the required valid components are available.

    Notes
    -----
    The sampling rectangle occupies 20% of total image area (each side
    spans the square root of 0.2 of its corresponding page dimension).
    After filtering character-shaped connected components, interquartile
    fences reject extreme heights. A 20--80% winsorized mean then reduces
    remaining outlier effects and median pixel-quantization artifacts.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    fraction = math.sqrt(.20)
    x0, x1 = round(w*(1-fraction)/2), round(w*(1+fraction)/2)
    y0, y1 = round(h*(1-fraction)/2), round(h*(1+fraction)/2)
    roi = gray[y0:y1, x0:x1]
    factor = min(1.0, 800/max(roi.shape))
    if factor < 1:
        roi = cv2.resize(roi, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    mask = foreground(roi)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    sizes = []
    for x, y, cw, ch, area in stats[1:]:
        if (2 <= cw <= roi.shape[1]*.075 and
            3 <= ch <= roi.shape[0]*.085 and
            4 <= area <= roi.size*.003 and
            .07 <= area/max(1,cw*ch) <= .9 and
            .12 <= cw/ch <= 3.0):
            sizes.append(float(ch)/factor)
    if len(sizes) < 25:
        return None
    sizes = np.array(sizes)
    q1, q3 = np.percentile(sizes, [25,75])
    sizes = sizes[(sizes >= max(3, q1*.65)) & (sizes <= q3*1.45)]
    if len(sizes) < 20:
        return None
    # A trimmed mean avoids whole-pixel quantization of median glyph height.
    a, b = np.percentile(sizes, [20, 80])
    return float(np.mean(np.clip(sizes, a, b)))


def half_letter_canvas(img, target_ratio=5.5/8.5, scale=1.):
    """Apply document-level font scale before final page placement.

    Parameters
    ----------
    img : numpy.ndarray
        Cropped and contrast-corrected RGB image.
    target_ratio : float, optional
        Historical compatibility parameter; currently unused. The paper
        aspect ratio is enforced by :func:`paste_centered`.
    scale : float, optional
        Isotropic resize factor, default 1.

    Returns
    -------
    numpy.ndarray
        RGB image resized by ``scale``; no page padding is added here.

    Notes
    -----
    Despite the historical name, this function does not create the actual
    half-letter canvas. The following placement step does that.
    """
    h, w = img.shape[:2]
    if abs(scale-1)>0.001:
        img = cv2.resize(img, (max(1,round(w*scale)), max(1,round(h*scale))),
                         interpolation=cv2.INTER_CUBIC if scale>1 else cv2.INTER_AREA)
    return img


def paste_centered(img, dpi, page_width=5.5, page_height=8.5):
    """Place an image centrally on a white half-letter raster canvas.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image to place, potentially after font-scale matching.
    dpi : int
        Output raster resolution in dots per inch.
    page_width : float, optional
        Physical page width in inches, default 5.5.
    page_height : float, optional
        Physical page height in inches, default 8.5.

    Returns
    -------
    canvas : numpy.ndarray
        RGB uint8 output at the prescribed physical page size.
    fit : float
        Additional uniform fit-down scale, never greater than 1.0.

    Notes
    -----
    Fitting never clips the input, but it can reduce final apparent font
    size if the selected crop exceeds the page canvas.
    """
    W, H = round(page_width*dpi), round(page_height*dpi)
    h, w = img.shape[:2]
    # If crop is too large, shrink *uniformly* rather than cut real text.
    fit = min(1.0, W/w, H/h)
    if fit < 1:
        w, h = max(1,round(w*fit)), max(1,round(h*fit))
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
    canvas = np.full((H,W,3),255,np.uint8)
    x, y = (W-w)//2, (H-h)//2
    canvas[y:y+h,x:x+w] = img
    return canvas, fit


def chapter_heading_component(img, lines, glyph):
    """Rescue one large chapter numeral positioned above a heading.

    Parameters
    ----------
    img : numpy.ndarray
        Deskewed RGB scan.
    lines : list of tuple
        Retained text-line rectangles and component counts.
    glyph : float or None
        Character-height estimate in source pixels.

    Returns
    -------
    list of tuple
        Either one detected chapter-glyph rectangle in line format, or an
        empty list when detection is ambiguous or prerequisites are absent.

    Notes
    -----
    This exception applies only to oversized heading numerals, *not* small
    isolated page numbers. It intentionally requires exactly one candidate.
    """
    if glyph is None or len(lines)<8:
        return []
    first=min(lines,key=lambda r:r[1])
    if first[1] < 3*glyph:
        return []
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    n,_,stats,_=cv2.connectedComponentsWithStats(foreground(gray),8)
    options=[]
    for xx,yy,ww,hh,area in stats[1:]:
        gap=first[1]-(yy+hh)
        if (2.5*glyph <= hh <= 5.5*glyph and
            0.45*glyph <= ww <= 3.5*glyph and
            4 <= gap <= 10*glyph and
            first[0]-7*glyph <= xx <= first[0]+8*glyph and
            0.10 <= area/max(1,ww*hh) <= .8):
            options.append((int(xx),int(yy),int(xx+ww),int(yy+hh),12))
    if len(options)==1:
        return options
    return []


def winsor_mean(values, trim=.20):
    """Calculate a robust mean by clipping both tails of the sample.

    Parameters
    ----------
    values : array_like
        Numeric observations, such as glyph heights or crop dimensions.
    trim : float, optional
        Fraction of the sample clipped at each tail, default 0.20.

    Returns
    -------
    float or None
        Mean after winsorization; None for an empty sample.

    Notes
    -----
    Winsorization does not discard observations; it limits their influence
    by replacing outlying values with the selected percentile thresholds.
    """
    vals = np.asarray(values, np.float64)
    if not len(vals):
        return None
    lo, hi = np.percentile(vals, [100*trim, 100*(1-trim)])
    return float(np.mean(np.clip(vals, lo, hi)))


def shared_crop_frame(rows, target, tolerance, dpi, margin_px):
    """Infer a consensus crop size from reliably comparable pages.

    Parameters
    ----------
    rows : list of dict
        Per-page analysis records from :func:`process_page`.
    target : float
        Document reference glyph height in pixels.
    tolerance : float
        Maximum relative glyph-height deviation allowed for a page to
        contribute to the shared frame, e.g. 0.08 for 8%.
    dpi : int
        Target output resolution, used to limit proposed crop dimensions.
    margin_px : int
        Compatibility parameter representing crop margin in pixels;
        not used directly by the current calculation.

    Returns
    -------
    tuple of int or None
        Shared ``(width, height)`` in reference-scale pixels, or None if
        fewer than three sufficiently text-rich pages are available.

    Notes
    -----
    Proposed crop sizes are first transformed into the common glyph scale.
    Their winsorized means make the frame less sensitive to a few erroneous
    bounding boxes. The frame is independently positioned for each page.
    """
    usable = [r for r in rows if r['glyph'] is not None and r['lines']
              and len(r['lines']) >= 10 and abs(r['glyph']/target-1) <= tolerance]
    if len(usable) < 3:
        return None
    widths = []
    heights = []
    for r in usable:
        x0,y0,x1,y1 = r['bbox']
        factor = target/r['glyph']
        widths.append((x1-x0)*factor)
        heights.append((y1-y0)*factor)
    # Never force overall document sizing to shrink the matched glyphs.
    max_w, max_h = round(5.5*dpi), round(8.5*dpi)
    return (max(1, min(max_w, round(winsor_mean(widths)))),
            max(1, min(max_h, round(winsor_mean(heights)))))


def optimize_window(lines, desired, dimension, direction):
    """Find the best placement of a fixed-size text-containing window.

    Parameters
    ----------
    lines : list of tuple
        Text-line rectangles and their component counts.
    desired : int
        Crop window length along the selected axis, in pixels.
    dimension : int
        Full image length along the selected axis, in pixels.
    direction : {0, 1}
        Axis selector: 0 for horizontal x bounds; 1 for vertical y bounds.

    Returns
    -------
    start : int
        Window starting coordinate in pixels.
    fraction : float
        Weighted fraction of text-line extents enclosed by the window.

    Notes
    -----
    Candidate positions are sampled every three pixels. When placements
    have approximately equal coverage, choose the more balanced one by
    proximity to the weighted center of the detected text.
    """
    if desired >= dimension:
        return 0, 1.0
    if not lines:
        return max(0,(dimension-desired)//2), 0.0
    left = np.asarray([r[direction] for r in lines],np.float64)
    right = np.asarray([r[direction+2] for r in lines],np.float64)
    widths = np.maximum(1.,right-left)
    weights = np.asarray([np.sqrt(max(1,r[4])) for r in lines],np.float64)
    # Probe all sensible crop positions at 3px resolution, including endpoints.
    starts = np.unique(np.r_[np.arange(0,dimension-desired+1,3),dimension-desired])
    intersect = np.maximum(0,np.minimum(right[None,:],(starts+desired)[:,None])-
                             np.maximum(left[None,:],starts[:,None]))
    fraction = np.minimum(1.,intersect/widths[None,:])
    scores = fraction@weights / weights.sum()
    optimum = np.max(scores)
    mid = np.average((left+right)/2,weights=weights)
    tie = np.flatnonzero(scores >= optimum-1e-6)
    idx = tie[np.argmin(abs(starts[tie]+desired/2-mid))]
    return int(starts[idx]),float(scores[idx])


def expand_crop_rect(rect, image_shape, fraction=0.05):
    """Expand a proposed crop 5% overall without crossing scan boundaries.

    Parameters
    ----------
    rect : tuple of int
        Original ``(x0, y0, x1, y1)`` crop rectangle.
    image_shape : tuple of int
        Shape of the original image; first two entries are height and width.
    fraction : float, optional
        Relative expansion of each crop dimension, default 0.05. Half of
        the expansion is applied to each side where space permits.

    Returns
    -------
    tuple of int
        Expanded, image-clamped bounding rectangle.
    """
    x0,y0,x1,y1 = rect
    h,w = image_shape[:2]
    extra_x = round((x1-x0)*fraction/2)
    extra_y = round((y1-y0)*fraction/2)
    return (max(0,x0-extra_x), max(0,y0-extra_y),
            min(w,x1+extra_x), min(h,y1+extra_y))


def document_crop(img, bbox, lines, frame, scale, pad=10):
    """Position a document-wide crop frame around each page's text.

    Parameters
    ----------
    img : numpy.ndarray
        Full deskewed RGB page.
    bbox : tuple of int
        Page-specific text-derived fallback bounds.
    lines : list of tuple
        Trusted text-line proposals in source coordinates.
    frame : tuple of int
        Desired common crop width/height in reference glyph-scale pixels.
    scale : float
        Page-to-reference glyph scaling factor; the desired frame is
        converted back to original coordinates before selecting a crop.
    pad : int, optional
        Safety padding when expanding a frame to avoid clipping, default 10.

    Returns
    -------
    cropped : numpy.ndarray
        RGB image region corresponding to the selected rectangle.
    rect : tuple of int
        Chosen original-image crop bounds ``(x0, y0, x1, y1)``.
    coverage : float
        Weighted detected-text coverage of the initial common frame.
    fallback : bool
        True if the window was expanded to avoid losing substantial text.

    Notes
    -----
    Horizontal and vertical placements are optimized independently.
    If the shared window covers less than 99.5% of weighted text evidence,
    or any substantial line is covered less than 96.5%, use a safety frame.
    """
    h,w=img.shape[:2]
    fw=min(w,max(1,round(frame[0]/scale)))
    fh=min(h,max(1,round(frame[1]/scale)))
    x,rx=optimize_window(lines,fw,w,0)
    y,ry=optimize_window(lines,fh,h,1)
    if lines:
        # Account for intersections in both directions, especially important
        # near chapter headings, line ends and the final paragraph.
        weight=0.
        kept=0.
        minimum=1.
        for x0,y0,x1,y1,count in lines:
            ww=max(1,x1-x0);hh=max(1,y1-y0)
            xr=max(0,min(x1,x+fw)-max(x0,x))/ww
            yr=max(0,min(y1,y+fh)-max(y0,y))/hh
            portion=xr*yr
            wt=math.sqrt(max(1,count))
            kept+=wt*portion
            weight+=wt
            if count >= 10:minimum=min(minimum,portion)
        coverage=kept/max(weight,1)
    else:
        coverage=0.;minimum=0.
    fallback=False
    if coverage < .995 or minimum < .965:
        # An isolated genuine heading or last line must not be cropped to make
        # the reference box fit. The fallback ignores noisy scan artifacts and
        # uses only the retained coherent line boxes.
        if lines:
            x0=max(0,min(r[0] for r in lines)-pad)
            x1=min(w,max(r[2] for r in lines)+pad)
            y0=max(0,min(r[1] for r in lines)-pad)
            y1=min(h,max(r[3] for r in lines)+pad)
            # Keep at least the reference frame whenever possible, centered on
            # genuine text rather than on a margin artifact.
            fw=min(w,max(fw,x1-x0)); fh=min(h,max(fh,y1-y0))
            x=max(0,min(w-fw,round((x0+x1-fw)/2)))
            y=max(0,min(h-fh,round((y0+y1-fh)/2)))
        else:
            x,y,x1,y1=bbox;fw=x1-x;fh=y1-y
        fallback=True
    return img[y:y+fh,x:x+fw],(x,y,x+fw,y+fh),coverage,fallback


def process_page(rgb, dpi, max_skew, margin_mm, tile, crop, text_detector="strict", text_filter="height-neighbors"):
    """Analyze one rasterized PDF page for later global crop decisions.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB uint8 source raster at the requested resolution.
    dpi : int
        Resolution used for rasterization.
    max_skew : float
        Maximum absolute deskew search angle, in degrees.
    margin_mm : float
        Safety margin around detected text in millimeters.
    tile : int
        Reserved for the subsequent normalization phase; unused here.
    crop : bool
        Whether to detect text bounds or retain the whole deskewed page.

    Returns
    -------
    dict
        Analysis record with ``image``, ``angle``, ``confidence``,
        ``glyph``, ``bbox``, ``lines``, and ``margin`` fields.

    Notes
    -----
    Glyph scale is always measured on the original deskewed image rather
    than on the crop proposal; incorrect crop bounds cannot skew the scale
    estimate. Actual cropping and enhancement happen in pass two.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    angle, confidence = estimate_skew(gray, max_skew)
    corrected = remove_copier_bands(rotate_bound(rgb, angle))
    margin_px = round(margin_mm / 25.4 * dpi)
    if crop:
        bbox,lines=text_bbox(corrected,margin_px,return_lines=True,detector=text_detector, spatial_filter=text_filter)
    else:
        h,w=corrected.shape[:2];bbox=(0,0,w,h);lines=[]
    # Measure only the center of the *original deskewed page* so that a bad
    # crop proposal cannot itself influence the font-scale measurement.
    measure = central_text_scale(corrected)
    if crop and lines:
        # Chapter numerals (e.g. a large "1") are part of a heading;
        # small isolated folio numbers are intentionally NOT recovered.
        lines=lines+chapter_heading_component(corrected,lines,measure)
    return dict(image=corrected,angle=angle,confidence=confidence,glyph=measure,
                bbox=bbox,lines=lines,margin=margin_px)



def main():
    """Run the two-pass command-line PDF cleanup pipeline.

    The first pass rasterizes and analyzes selected pages without committing
    to a crop. The second pass computes document-wide reference values,
    chooses page-specific or shared crops, normalizes contrast, fits the
    content to half-letter pages, and writes the image-only PDF.

    Returns
    -------
    None
        Writes an output PDF and per-page diagnostic messages to stderr.

    Notes
    -----
    The PDF is rasterized. Existing searchable text layers and vector
    elements are not carried over to the output.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--dpi', type=int, default=320,
                    help='output raster resolution in DPI (default: 320; avoids reducing common high-resolution scans)')
    ap.add_argument('--max-skew', type=float, default=8.0)
    ap.add_argument('--margin-mm', type=float, default=7.0)
    ap.add_argument('--tile-ems', type=float, default=4.0,
                    help='local white calibration cell side in document glyph heights (default: 4; clamped to 2%% of shortest and 20%% of longest image dimension)')
    ap.add_argument('--pages', default=None, help='page range, e.g. 1-3 or 2,4,6')
    ap.add_argument('--no-crop', action='store_true', help='keep whole page after straightening')
    ap.add_argument('--font-tolerance', type=float, default=.08,
                    help='normalize pages whose estimated glyph size differs at most this fraction from median')
    ap.add_argument('--no-font-match', action='store_true', help='disable font-size matching')
    ap.add_argument('--no-shared-crop', action='store_true',help='disable document-wide common crop frame')
    ap.add_argument('--remove-vertical-artifacts', action='store_true',
                    help='remove persistent vertical edge/spine shadows outside detected text')
    ap.add_argument('--remove-slanted-edge-artifacts', action='store_true',
                    help='trace slanted edge/spine shadows, whiten toward page edges while protecting text')
    ap.add_argument('--jpeg-quality', type=int, default=72,
                    help='grayscale JPEG quality 1-100 (default: 72; smaller PDF than PNG)')
    ap.add_argument('--text-feather-fonts', type=float, default=0.8,
                    help='Gaussian feather sigma in units of average output glyph height (default: 0.8)')
    ap.add_argument('--contrast-strength', type=float, default=5.0,
                    help='text black-clipping multiplier (default: 5; 1 = ink peak goes to black)')
    ap.add_argument('--max-empty-lines', type=float, default=1.0,
                    help='maximum empty text lines bridged by one contrast rectangle (default: 1; inf for one box)')
    ap.add_argument('--text-detector', choices=['legacy','strict','adaptive','sauvola','background'],
                    default='strict',help='raw pixel detector used for text-line proposals')
    ap.add_argument('--text-filter', choices=['off','neighbors','baseline','combined','tuned-baseline','tuned-neighbors','height-neighbors','height-neighbors-rescue'],
                    default='height-neighbors', help='spatial pixel filter before text-line grouping')
    ap.add_argument('--text-mask-method', choices=['median','mean','maximum','quantile'], default='median',
                    help='horizontal block width statistic for text contrast mask')
    ap.add_argument('--no-text-mask-outliers', action='store_true',
                    help='omit original line rectangles outside the grouped block')
    ap.add_argument('--text-contrast', action='store_true',
                    help='use stronger contrast near detected text lines')
    ap.add_argument('--white-strength', type=float, default=1.0,
                    help='global paper-white clipping strength, independent of ink contrast (default: 1)')
    ap.add_argument('--background-contrast', type=float, default=0.0,
                    help='black clipping multiplier outside text (default: 0)')
    args = ap.parse_args()
    if args.text_feather_fonts < 0 or not math.isfinite(args.text_feather_fonts): ap.error('--text-feather-fonts must be nonnegative and finite')
    if not math.isfinite(args.tile_ems) or args.tile_ems <= 0: ap.error('--tile-ems must be positive and finite')
    if not 1 <= args.jpeg_quality <= 100: ap.error('--jpeg-quality must be 1..100')
    if not math.isfinite(args.white_strength) or args.white_strength < 0: ap.error('--white-strength must be >= 0')
    if not math.isfinite(args.contrast_strength) or args.contrast_strength < 0: ap.error('--contrast-strength must be finite and >= 0')
    if not math.isfinite(args.background_contrast) or args.background_contrast < 0: ap.error('--background-contrast must be finite and >= 0')
    if args.input.resolve() == args.output.resolve():
        ap.error('Input and output must differ')
    source = fitz.open(args.input)
    output = fitz.open()
    if args.pages:
        indices = []
        for term in args.pages.split(','):
            if '-' in term:
                lo, hi = map(int, term.split('-', 1))
                indices.extend(range(lo-1, hi))
            else:
                indices.append(int(term)-1)
        if any(i < 0 or i >= len(source) for i in indices):
            ap.error('page number out of range')
    else:
        indices = list(range(len(source)))
    # PASS 1: gather independent geometry/scale evidence from each original
    # page. It is important not to use an early crop to estimate glyph scale,
    # because a bad crop would then influence how that same page is resized.
    # Two passes: measure glyphs + text blocks before deciding shared crop.
    processed = []
    for i in indices:
        page = source[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(args.dpi/72,args.dpi/72),
                              colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height,pix.width,3).copy()
        row=process_page(img,args.dpi,args.max_skew,args.margin_mm,None,not args.no_crop,
                         text_detector=args.text_detector, text_filter=args.text_filter)
        row['index']=i
        processed.append(row)
    valid = [r['glyph'] for r in processed if r['glyph'] is not None]
    # Use the trimmed mean, not a single potentially unrepresentative page.
    target = winsor_mean(valid) if valid else None
    print(f'Central glyph reference: {target:.2f} pixels' if target else
          'Central glyph reference unavailable; no font matching', file=sys.stderr)
    frame=None
    if target and not args.no_crop and not args.no_font_match and not args.no_shared_crop:
        frame=shared_crop_frame(processed,target,args.font_tolerance,args.dpi,
                                round(args.margin_mm / 25.4 * args.dpi))
        if frame:
            print(f'Shared crop frame at reference glyph scale: {frame[0]} x {frame[1]} pixels',file=sys.stderr)
    # PASS 2: select either a consensus crop or a conservative per-page crop,
    # then perform enhancement and render onto a fixed-size white canvas.
    for row in processed:
        i=row['index']
        angle=row['angle'];confidence=row['confidence'];measure=row['glyph']
        ratio = measure/target if (measure is not None and target is not None) else None
        match = (not args.no_font_match and ratio is not None and
                 len(row['lines']) >= 10 and
                 abs(ratio-1) <= args.font_tolerance)
        factor = target/measure if match else 1.0
        mode='native'; coverage=1.
        if frame is not None and match:
            result,rect,coverage,fallback=document_crop(
                row['image'],row['bbox'],row['lines'],frame,factor,pad=round(args.dpi*.10))
            mode='safety-expand' if fallback else 'shared'
        else:
            x0,y0,x1,y1=row['bbox'];result=row['image'][y0:y1,x0:x1]
            rect=row['bbox']
        # Sparse title pages are particularly vulnerable to false-negative line
        # detections. Keep the full scan on these pages instead of interpreting
        # one missing wide heading as a safe reason to crop through its letters.
        # Do not weaken the shared crop of normal, densely printed body pages.
        if mode == 'native' and len(row['lines']) < 18:
            hh, ww = row['image'].shape[:2]
            rect = (0, 0, ww, hh)
            result = row['image']
            mode = 'sparse-full-page'
        # This expansion happens *after* the shared-crop safety test, so every
        # crop—including per-page fallbacks—receives the same extra whitespace.
        # Expand every final crop by 5% in both dimensions (2.5% per side).
        rect = expand_crop_rect(rect, row['image'].shape, fraction=0.05)
        x0,y0,x1,y1 = rect
        result = row['image'][y0:y1,x0:x1]
        # In particular, do not clip the corrected background at the detected
        # top/bottom text limits: doing so caused the abrupt artificial white
        # boundary seen in the previous version of page 4.
        # Do not white-out horizontal strips inside a crop. The continuous
        # paper-background correction below handles those areas instead.
        if args.remove_vertical_artifacts and mode != 'sparse-full-page':
            # Kept in an independent module so the established crop/contrast
            # algorithm remains unchanged and this option can be disabled.
            from remove_vertical_scan_artifacts import remove_vertical_edge_artifacts
            crop_lines = [(max(0, a-x0), max(0, b-y0),
                           min(x1-x0, c-x0), min(y1-y0, d-y0), count)
                          for a,b,c,d,count in row['lines']
                          if c > x0 and a < x1 and d > y0 and b < y1]
            result = remove_vertical_edge_artifacts(result, text_lines=crop_lines)
        if args.remove_slanted_edge_artifacts and mode != 'sparse-full-page':
            # Optional independent stage: the legacy filter above remains
            # untouched. Prefer this tracer for tapered / slanted shadows.
            from remove_slanted_edge_artifacts import remove_slanted_edge_artifacts
            if not args.remove_vertical_artifacts:
                crop_lines = [(max(0, a-x0), max(0, b-y0),
                               min(x1-x0, c-x0), min(y1-y0, d-y0), count)
                              for a,b,c,d,count in row['lines']
                              if c > x0 and a < x1 and d > y0 and b < y1]
            result = remove_slanted_edge_artifacts(result, text_lines=crop_lines)
        # Document-wide glyph scale was computed before enhancement in pass one.
        # Convert from source page to crop raster units (both use source DPI).
        grid_pixels = grid_cell_pixels(result.shape, target or measure or 24.0, args.tile_ems)
        result=normalize_local(result,tile=grid_pixels)
        scaled=half_letter_canvas(result,scale=factor)
        finished,fit=paste_centered(scaled,args.dpi)
        # The final PDF stores one 8-bit *grayscale* JPEG per page. Converting
        # after local enhancement preserves the existing detection/cleanup code
        # while reducing the output from three color channels to one.
        gray_final = cv2.cvtColor(finished, cv2.COLOR_RGB2GRAY)
        # Map the already detected text lines through crop, font scaling,
        # and final fit-down into finished-page pixel coordinates. Geometry
        # and crop decisions remain unchanged.
        text_reference = None
        if args.text_contrast and row['lines']:
            h0, w0 = result.shape[:2]
            scaled_h, scaled_w = scaled.shape[:2]
            actual_x = scaled_w / w0 * fit
            actual_y = scaled_h / h0 * fit
            out_h, out_w = gray_final.shape
            placed_w = round(scaled_w * fit) if fit < 1 else scaled_w
            placed_h = round(scaled_h * fit) if fit < 1 else scaled_h
            offset_x = (out_w - placed_w)//2
            offset_y = (out_h - placed_h)//2
            final_lines = [((a-x0)*actual_x+offset_x,
                            (b-y0)*actual_y+offset_y,
                            (c-x0)*actual_x+offset_x,
                            (d-y0)*actual_y+offset_y)
                           for a,b,c,d,*_ in row['lines']]
            # Glyph measurements were made in deskewed source pixels. Convert
            # them to final output pixels before specifying feather distance.
            glyph_output = ((measure if measure is not None else (target or 24.0))
                            * actual_y)
            feather_pixels = args.text_feather_fonts * glyph_output
            text_reference = text_contrast_mask(gray_final.shape, final_lines,
                                                background=0.0, foreground=1.0,
                                                feather=feather_pixels,
                                                max_empty_lines=args.max_empty_lines,
                                                method=args.text_mask_method,
                                                include_outliers=not args.no_text_mask_outliers)
            strength = text_contrast_mask(gray_final.shape, final_lines,
                                          background=args.background_contrast,
                                          foreground=args.contrast_strength,
                                          feather=feather_pixels,
                                          max_empty_lines=args.max_empty_lines,
                                          method=args.text_mask_method,
                                          include_outliers=not args.no_text_mask_outliers)
        else:
            strength = args.contrast_strength
        gray_final = enhance_grayscale(gray_final, strength=strength,
                                        white_strength=args.white_strength,
                                        text_reference=text_reference)

        encoded = encode_grayscale_jpeg(gray_final, quality=args.jpeg_quality)
        new=output.new_page(width=5.5*72,height=8.5*72)
        new.insert_image(new.rect,stream=encoded)
        print(f'page {i+1:3}: rotation {angle:+.2f}, confidence {confidence:.2f}, '
              f'glyph {measure if measure else 0:.1f}px, match x{factor:.3f}, '
              f'crop {mode} (text {coverage:.1%}), fit x{fit:.3f}',file=sys.stderr)
    output.save(args.output,garbage=4,deflate=True)
    output.close()
    source.close()
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
