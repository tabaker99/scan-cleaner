"""Pixel-level ink detection for scanned book pages.

These methods intentionally do NOT create text-line rectangles or broaden the
mask to fill line gaps; those are independent, later geometry operations.
"""
import cv2
import numpy as np


def detect_ink(gray, method='legacy'):
    """Return a uint8 (0/1) raw candidate-ink mask.

    Parameters
    ----------
    gray : numpy.ndarray
        Single-channel uint8 unenhanced, deskewed scan.
    method : {'legacy', 'adaptive', 'sauvola', 'background'}
        Alternative binarization approaches. Defaults to ``legacy`` for
        compatibility with the existing crop algorithm. ``adaptive`` uses
        adaptive Gaussian thresholds to recover low-contrast strokes;
        ``sauvola`` handles nonuniform paper background; ``background``
        subtracts a smooth morphological illumination estimate.

    Returns
    -------
    numpy.ndarray
        Binary candidate-ink mask without line extension, dilation, or OCR.
    """
    if gray.ndim!=2 or gray.dtype!=np.uint8:
        raise ValueError('gray must be 2D uint8')
    if method=='legacy':
        small=cv2.resize(gray,None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA)
        bg=cv2.morphologyEx(small,cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))
        bg=cv2.resize(bg,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_CUBIC)
        return ((bg.astype(np.int16)-gray.astype(np.int16))>28).astype(np.uint8)
    if method=='strict':
        # More selective background subtraction: removes faint scan texture.
        small=cv2.resize(gray,None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA)
        bg=cv2.morphologyEx(small,cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))
        bg=cv2.resize(bg,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_CUBIC)
        return ((bg.astype(np.int16)-gray.astype(np.int16))>36).astype(np.uint8)
    if method=='adaptive':
        smooth=cv2.GaussianBlur(gray,(3,3),0)
        return (cv2.adaptiveThreshold(smooth,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,39,14)>0).astype(np.uint8)
    if method=='sauvola':
        f=gray.astype(np.float32)
        mean=cv2.boxFilter(f,-1,(39,39),normalize=True)
        variance=cv2.boxFilter(f*f,-1,(39,39),normalize=True)-mean*mean
        sd=np.sqrt(np.maximum(0,variance))
        return (f<mean*(1+.23*(sd/128-1))).astype(np.uint8)
    if method=='background':
        small=cv2.resize(gray,None,fx=.35,fy=.35,interpolation=cv2.INTER_AREA)
        bg=cv2.morphologyEx(small,cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
        bg=cv2.resize(bg,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_CUBIC)
        return ((bg.astype(np.int16)-gray.astype(np.int16))>20).astype(np.uint8)
    raise ValueError(f'Unknown ink detector: {method}')
