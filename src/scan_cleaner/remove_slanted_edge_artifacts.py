"""Trace and remove slanted scanner shadows along page edges.

Designed for an already deskewed, cropped RGB page before contrast adjustment.
This independent module does not alter the existing deskew/crop pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np


def remove_slanted_edge_artifacts(
    rgb: np.ndarray,
    text_lines: list[tuple] | None = None,
    edge_fraction: float = 0.22,
    safety_fraction: float = 0.012,
    darkness_threshold: float = 0.13,
    return_masks: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Whiten edge-connected shadows, including their slanted continuations.

    Parameters
    ----------
    rgb : ndarray of uint8, shape (height, width, 3)
        Cropped RGB image, prior to the final local contrast normalization.
    text_lines : list of tuples, optional
        Trusted text line bounds ``(x0, y0, x1, y1, component_count)``.
        A safety corridor around the global horizontal text bounds is never
        modified. If missing, the central 70 percent remains protected.
    edge_fraction : float, default 0.22
        Maximum horizontal search extent from either outer edge.
    safety_fraction : float, default 0.012
        Extra gap between detected text and the shadow-removal mask, as a
        fraction of image width.
    darkness_threshold : float, default 0.13
        Minimum average dark coverage per vertical stripe/column. Higher
        values make detection more conservative.
    return_masks : bool, default False
        Also return the binary whitening mask for diagnostics.

    Returns
    -------
    ndarray or tuple of (ndarray, ndarray)
        Whitened image and, optionally, its uint8 mask.

    Notes
    -----
    Each side is treated independently. The method samples horizontal bands,
    estimates dark edge-shadow extents using an average darkness statistic,
    and fits a robust first-order model to the edge-facing boundary. The fitted
    model propagates a coherent slant through regions where the band becomes
    too faint to satisfy the threshold locally. A detected column segment is
    removed all the way to the image edge, provided it lies entirely outside
    protected text margins.

    This assumes ordinary printed content lies inside the trusted text bounds.
    Decorations or marginalia outside those bounds can be removed.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected RGB uint8 image with shape (H, W, 3)')
    h, w = rgb.shape[:2]
    if h < 90 or w < 100:
        return (rgb.copy(), np.zeros((h,w), np.uint8)) if return_masks else rgb.copy()
    if text_lines:
        valid = [line for line in text_lines if line[2] > line[0] and line[3] > line[1]]
    else:
        valid = []
    if valid:
        left = min(float(r[0]) for r in valid)
        right = max(float(r[2]) for r in valid)
    else:
        left, right = w*.15, w*.85
    pad = max(4, round(w*safety_fraction))
    safe_left = max(0, int(left)-pad)
    safe_right = min(w, int(right)+pad)
    max_extent = min(round(edge_fraction*w), round(w*.32))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Anti-alias camera noise but do not merge nearby characters into bars.
    gray = cv2.GaussianBlur(gray, (3,3), .7)
    darkness = np.clip((247-gray)/55., 0, 1)
    stripe_h = max(18, round(h/42))
    centers = np.arange(stripe_h//2, h, stripe_h, dtype=np.float64)
    mask = np.zeros((h,w), dtype=np.uint8)
    for side in ('left', 'right'):
        lim = min(max_extent, safe_left if side=='left' else w-safe_right)
        if lim < 4:
            continue
        # Coordinates are outward -> inward on both sides.
        slab = darkness[:, :lim] if side=='left' else darkness[:, ::-1][:, :lim]
        extent = np.full(len(centers), np.nan, np.float64)
        strength = np.zeros(len(centers), np.float64)
        for j,cy in enumerate(centers):
            ya=max(0,int(cy-stripe_h*.7)); yb=min(h,int(cy+stripe_h*.7))
            profile=slab[ya:yb].mean(axis=0)
            profile=cv2.GaussianBlur(profile[None].astype(np.float32),(0,0),sigmaX=max(1, w*.001))[0]
            # Treat only sustained patches as shading; text-like speckles
            # rarely cover enough of the stripe to meet this threshold.
            dark=profile>darkness_threshold
            # Reject tiny isolated dark pieces using 1-D opening.
            k=max(2,round(w*.002))
            dark=cv2.morphologyEx(dark.astype(np.uint8)[None], cv2.MORPH_CLOSE,
                                   np.ones((1,k),np.uint8))[0].astype(bool)
            starts=np.flatnonzero(np.diff(np.r_[False,dark,False].astype(np.int8)))
            runs=[(a,b) for a,b in zip(starts[::2],starts[1::2]) if b-a>=max(2,k)]
            if not runs:
                continue
            # Choose runs close enough to the edge to be scanning artifacts.
            # A side shadow may be inset slightly with white paper outside it.
            runs=[(a,b) for a,b in runs if a < max(12,lim*.60)]
            if not runs:
                continue
            a,b=max(runs,key=lambda r:(r[1]-r[0]) * float(np.mean(profile[r[0]:r[1]])))
            extent[j]=float(b+max(2,round(w*.004)))
            strength[j]=float(np.mean(profile[a:b]))

        good=np.isfinite(extent)
        mid=(centers>h*.17)&(centers<h*.83)
        # Require repeated central evidence; an isolated footer/scan blotch
        # is insufficient to qualify as a page-long spine shadow.
        if np.count_nonzero(good&mid)<max(3,round(np.count_nonzero(mid)*.18)):
            continue
        # Fit line to visible locations, suppressing outlier widths from
        # short horizontal marks and occasional text-like artifact texture.
        yy=centers[good]; xx=extent[good]
        if len(xx)>4:
            slope, intercept=np.polyfit(yy,xx,1)
            residual=xx-(slope*yy+intercept)
            med=np.median(residual)
            tol=max(7.,2.8*1.4826*np.median(np.abs(residual-med)))
            inliers=np.abs(residual-med)<=tol
            if inliers.sum()>=3:
                slope, intercept=np.polyfit(yy[inliers],xx[inliers],1)
        else:
            slope=0.;intercept=float(np.median(xx))
        ys=np.arange(h,dtype=float)
        modeled=slope*ys+intercept
        # Suppress unreasonably aggressive extrapolation beyond observed band.
        lower=max(1, np.percentile(xx,15)-max(5,w*.01))
        upper=min(lim,np.percentile(xx,85)+max(8,w*.022))
        modeled=np.clip(modeled,lower,upper)
        # Extend the band over all y, even where it fades out: this is the
        # key difference from the old full-column detector.
        widths=np.rint(np.clip(modeled,0,lim)).astype(int)
        for y,n in enumerate(widths):
            if side=='left':
                mask[y,:n]=255
            else:
                mask[y,w-n:]=255
    result=rgb.copy()
    result[mask>0]=255
    return (result,mask) if return_masks else result
