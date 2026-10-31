#!/usr/bin/env python3
"""Post-crop image enhancement and compressed grayscale PDF image encoding.

This module never determines crop bounds or changes page geometry. The main
cleaner imports it only after text geometry has already been measured.
"""
import math
import cv2
import numpy as np

def grid_cell_pixels(shape, document_glyph_height, cell_ems=4.0):
    """Calculate square local-calibration cell size from document glyph scale.

    Parameters
    ----------
    shape : tuple
        Image shape (height, width[, channels]) at the processing stage.
    document_glyph_height : float
        Document-wide representative glyph height in pixels *at this stage*.
    cell_ems : float, optional
        Cell width in representative glyph heights; default 4.

    Returns
    -------
    int
        Approximate square cell side, clamped to [2% of shorter dimension,
        20% of longer dimension]. The grid samples are evenly partitioned.
    """
    h, w = shape[:2]
    if not np.isfinite(document_glyph_height) or document_glyph_height <= 0:
        raise ValueError('document_glyph_height must be positive and finite')
    if not np.isfinite(cell_ems) or cell_ems <= 0:
        raise ValueError('cell_ems must be positive and finite')
    lower = max(1, math.ceil(0.02 * min(h, w)))
    upper = max(lower, math.floor(0.20 * max(h, w)))
    return int(np.clip(round(cell_ems * document_glyph_height), lower, upper))


def normalize_local(rgb, tile=128, white_target=255):
    """Flatten uneven paper illumination using a softly interpolated grid.

    The local step intentionally adjusts only the paper white reference.
    It does NOT stretch dark pixels to black: the later independently masked
    histogram-peak stage handles ink clipping, preventing double correction.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB image after crop and edge-artifact cleanup.
    tile : int
        Approximate square-grid side length in pixels. Call
        :func:`grid_cell_pixels` to select it from document em units.
    white_target : int
        Desired paper white (255).

    Returns
    -------
    numpy.ndarray
        uint8 RGB image with smoothed local white-point correction.
    """
    if tile < 1:
        raise ValueError('tile must be positive')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    ny, nx = max(2, math.ceil(h/tile)), max(2, math.ceil(w/tile))
    white_grid = np.empty((ny, nx), np.float32)
    for j in range(ny):
        for i in range(nx):
            patch = gray[j*h//ny:(j+1)*h//ny, i*w//nx:(i+1)*w//nx]
            white_grid[j, i] = float(np.percentile(patch, 90))
    white_grid = cv2.GaussianBlur(white_grid, (0,0), 0.9)
    white = cv2.resize(white_grid, (w,h), interpolation=cv2.INTER_CUBIC)
    # Subtractive correction preserves local stroke separation and ink tones.
    correction = np.maximum(0.0, white_target - white)
    out = np.clip(rgb.astype(np.float32) + correction[...,None], 0,255).astype(np.uint8)
    # Paper noise already near white is removed before JPEG compression.
    out[np.all(out >= 248, axis=2)] = 255
    return out


def histogram_peaks(gray, text_weight=None):
    """Estimate ink and paper modes from a smoothed grayscale histogram.

    A text weighting map restricts dark-peak estimation to likely text. The
    overall paper mode is estimated from the full image. Returns None for the
    ink mode when the dark cluster is insufficiently represented.
    """
    from scipy.ndimage import gaussian_filter1d
    image = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    smooth = gaussian_filter1d(hist, 3.0)
    # Search the brighter half of the observed paper distribution.
    paper_lo = max(150, int(np.percentile(image, 55)))
    white_peak = int(paper_lo + np.argmax(smooth[paper_lo:]))
    cutoff = min(190, white_peak-40)
    if cutoff < 30:
        return None, float(white_peak)
    if text_weight is None:
        pixels = image.ravel()
    else:
        mask = np.asarray(text_weight)
        if mask.shape != image.shape:
            raise ValueError('text_weight must match image dimensions')
        pixels = image[mask > 0.55]
    dark = pixels[(pixels >= 8) & (pixels < cutoff)]
    if dark.size < 30:
        return None, float(white_peak)
    dark_hist = np.bincount(dark, minlength=256).astype(np.float64)
    dark_smooth = gaussian_filter1d(dark_hist, 4.0)
    # A dark peak must represent multiple pixels rather than a single speck.
    peak = int(np.argmax(dark_smooth[:cutoff]))
    if dark.size < max(30, pixels.size * 0.0002) or white_peak-peak < 45:
        return None, float(white_peak)
    return float(peak), float(white_peak)


def enhance_grayscale(gray, strength=1.0, white_strength=1.0,
                      text_reference=None, return_peaks=False):
    """Apply linear clipping with separately controlled black and white limits.

    Parameters
    ----------
    gray : numpy.ndarray
        2D uint8 grayscale image.
    strength : nonnegative float or numpy.ndarray
        Black-point aggression: 0 avoids black enhancement, 1 maps the ink
        histogram peak to black, and >1 pushes the clipping boundary beyond
        that peak. Pixel masks allow different contrast in text/background.
    white_strength : nonnegative float
        White-point clipping independent of black enhancement. Default 1
        moves the paper peak slightly inside the white clipping threshold.
    text_reference : numpy.ndarray, optional
        Pixel-aligned [0,1] text likelihood mask for estimating the black
        peak without scanner edges.
    return_peaks : bool
        If true, also return diagnostic (black_peak, white_peak).

    Returns
    -------
    numpy.ndarray or tuple
        Grayscale image, optionally accompanied by peak positions.

    Notes
    -----
    Purely linear clipping: no gamma, unsharp mask, or tone-curve exponent.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('gray must be a two-dimensional uint8 image')
    if np.isscalar(strength):
        strength = float(strength)
        if not np.isfinite(strength) or strength < 0:
            raise ValueError('strength must be finite and nonnegative')
    else:
        strength = np.asarray(strength, dtype=np.float32)
        if strength.shape != gray.shape or not np.all(np.isfinite(strength)) or np.any(strength<0):
            raise ValueError('strength mask must match image and contain finite nonnegative values')
    if not np.isfinite(white_strength) or white_strength < 0:
        raise ValueError('white_strength must be finite and nonnegative')
    black, white = histogram_peaks(gray, text_reference)
    original = gray.astype(np.float32)
    # White clipping applies everywhere, including non-text, to remove paper.
    white_cut = white - min(float(white_strength), 3.0) * 0.035 * (255.0-white+50.0)
    white_cut = max(white-25.0, min(255.0,white_cut))
    # No reliable dark peak? Do not invent one or amplify dark artifacts.
    if black is None:
        black_cut = 0.0
    else:
        # At s=1 black peak maps to zero. Values >1 clip increasingly lighter
        # inks, up to a bounded extra fraction of ink-to-paper separation.
        extra = np.clip(strength-1.0,0.0,4.0)*0.075*(white-black)
        black_cut = black*np.minimum(strength,1.0) + extra
    black_cut = np.minimum(black_cut, white_cut-32.0)
    out = np.clip((original-black_cut)*(255.0/np.maximum(white_cut-black_cut,1.0)),0,255)
    result = np.rint(out).astype(np.uint8)
    return (result,(black,white)) if return_peaks else result


def expand_text_regions(lines, max_empty_lines=1.0, method="median",
                        include_outliers=True, quantile=0.15):
    """Group successive text lines into robust horizontal rectangular blocks.

    Parameters
    ----------
    lines : iterable of tuple
        Original (x0, y0, x1, y1, ...) line proposals, before feathering.
    max_empty_lines : float or None
        Maximum skipped line pitches before starting another block; ``None``
        joins all text into a single block.
    method : {'median', 'mean', 'maximum', 'quantile'}
        How to select the *horizontal* block edges. ``maximum`` uses the
        extrema of the line rectangles; median/mean/quantile reduce the
        influence of rare edge-blotches. Vertical edges always use extremes.
    include_outliers : bool
        When true, also return individual line rectangles alongside the
        common block, so an unusually long legitimate line isn't lost.
    quantile : float
        Tail probability for the ``quantile`` method (default 0.15).

    Returns
    -------
    list[tuple[int, int, int, int]]
        Block rectangles, followed by original rectangles when requested.
        Duplicates are removed. Padding and smoothing are *not* applied.
    """
    if method not in ('median', 'mean', 'maximum', 'quantile'):
        raise ValueError('method must be median, mean, maximum, or quantile')
    if not 0 <= quantile < 0.5:
        raise ValueError('quantile must be in [0, 0.5)')
    if max_empty_lines is not None and (not np.isfinite(max_empty_lines) and not np.isinf(max_empty_lines) or max_empty_lines < 0):
        raise ValueError('max_empty_lines must be nonnegative or None')
    rectangles=[]
    for line in lines:
        x0,y0,x1,y1=(int(round(v)) for v in line[:4])
        if x1>x0 and y1>y0:
            rectangles.append((x0,y0,x1,y1))
    if not rectangles:
        return []
    rectangles.sort(key=lambda r:(r[1],r[0]))
    groups=[]
    if max_empty_lines is None or np.isinf(max_empty_lines):
        groups=[rectangles]
    else:
        heights=np.array([r[3]-r[1] for r in rectangles],np.float64)
        typical_height=float(np.median(heights))
        centers=np.array([(r[1]+r[3])/2 for r in rectangles],np.float64)
        diffs=np.diff(centers)
        good=diffs[(diffs >= .85*typical_height)&(diffs <= 2.2*typical_height)]
        pitch=float(np.median(good)) if good.size else 1.5*typical_height
        maximum_step=(1+float(max_empty_lines))*pitch+.25*typical_height
        group=[rectangles[0]]
        for r,center,prev in zip(rectangles[1:],centers[1:],centers[:-1]):
            if center-prev > maximum_step:
                groups.append(group)
                group=[]
            group.append(r)
        groups.append(group)
    result=[]
    for group in groups:
        left=np.array([r[0] for r in group],np.float64)
        right=np.array([r[2] for r in group],np.float64)
        if method=='maximum':
            xl,xr=float(left.min()),float(right.max())
        elif method=='median':
            xl,xr=float(np.median(left)),float(np.median(right))
        elif method=='mean':
            xl,xr=float(left.mean()),float(right.mean())
        else:
            xl,xr=float(np.quantile(left,quantile)),float(np.quantile(right,1-quantile))
        block=(int(np.floor(xl)),min(r[1] for r in group),
               int(np.ceil(xr)),max(r[3] for r in group))
        if block[2]>block[0]:
            result.append(block)
        if include_outliers:
            # Preserve the original actual line boxes as a *union*, rather
            # than increasing the common rectangle across every scan defect.
            result.extend(group)
    return list(dict.fromkeys(result))


def text_contrast_mask(shape, lines, background=0.3, foreground=1.0,
                       padding=8, feather=17, max_empty_lines=1.0,
                       method='median', include_outliers=True, quantile=0.15):
    """Create a float contrast-strength map from robust grouped text boxes.

    The method parameter only changes the horizontal common block width;
    outlier retention optionally unions original line rectangles with blocks.
    Both take effect *before* padding and Gaussian feathering.
    """
    h,w=shape[:2]
    if not (np.isfinite(background) and np.isfinite(foreground)
            and background>=0 and foreground>=0):
        raise ValueError('contrast multipliers must be finite and nonnegative')
    if padding<0 or feather<0:
        raise ValueError('padding and feather must be nonnegative')
    rectangles=expand_text_regions(lines,max_empty_lines=max_empty_lines,
                                   method=method,include_outliers=include_outliers,
                                   quantile=quantile)
    mask=np.zeros((h,w),np.uint8)
    for x0,y0,x1,y1 in rectangles:
        xa,ya=max(0,x0-padding),max(0,y0-padding)
        xb,yb=min(w,x1+padding),min(h,y1+padding)
        if xb>xa and yb>ya:
            mask[ya:yb,xa:xb]=255
    if feather:
        mask=cv2.GaussianBlur(mask,(0,0),sigmaX=feather,sigmaY=feather)
    return (background+(foreground-background)*mask.astype(np.float32)/255).astype(np.float32)


def encode_grayscale_jpeg(gray, quality=72):
    """Encode an 8-bit grayscale page as optimized JPEG bytes.

    Parameters
    ----------
    gray : numpy.ndarray
        Single-channel uint8 page.
    quality : int, optional
        JPEG quality, default 72.

    Returns
    -------
    bytes
        JPEG stream suitable for ``fitz.Page.insert_image``.
    """
    if gray.ndim != 2:
        raise ValueError('Expected single-channel grayscale image')
    ok, encoded = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, int(quality),
                                               cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        raise RuntimeError('Grayscale JPEG encoding failed')
    return encoded.tobytes()
