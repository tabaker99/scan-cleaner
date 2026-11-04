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


def detect_stroke_cores(gray, low_floor=29, local_fraction=0.58,
                        local_window=7, strong_threshold=62):
    """Identify ink interiors with local relative contrast and strong-pixel support.

    Separates ink from bright antialiasing halos while permitting weaker strokes
    when their *local* intensity drop remains meaningful. Pixel inclusion uses
    the illumination-corrected darkness relative to the strongest nearby ink,
    rather than the darkness of the darkest portion of an entire character.

    Parameters
    ----------
    gray : ndarray
        Source grayscale page, before enhancement.
    low_floor : int
        Minimum darkness relative to local paper for any retained pixel.
    local_fraction : float
        Minimum ink darkness as a fraction of the strongest ink nearby.
    local_window : int
        Window width used to compare within a small stroke neighborhood.
    strong_threshold : int
        Absolute threshold establishing reliably dark ink seed regions.

    Returns
    -------
    ndarray
        Binary mask of detected ink interiors. Original pixels are unchanged.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('gray must be 2D uint8')
    small=cv2.resize(gray,None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA)
    paper=cv2.morphologyEx(small,cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))
    paper=cv2.resize(paper,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_CUBIC)
    drop=np.maximum(0,paper.astype(np.float32)-gray.astype(np.float32))
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(local_window,local_window))
    local_peak=cv2.dilate(drop,k)
    # The extra local fraction is what removes faint fringes bordering dark ink.
    candidate=(drop>=low_floor)&(drop>=local_fraction*local_peak)
    # An isolated gray paper speck can satisfy the relative test; require a
    # connected dark seed, or a locally credible smaller stroke with structure.
    strong=drop>=strong_threshold
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8),8)
    seed_labels=np.unique(labels[strong & candidate])
    accepted=np.zeros(count,np.uint8)
    accepted[seed_labels]=1
    accepted[0]=0
    # For faint, completely disconnected strokes, allow coherent groups of
    # multiple pixels with moderate local contrast, not 1-pixel noise.
    sizes=np.bincount(labels.ravel(),minlength=count)
    local_seed=(drop>=44)&candidate
    weak_ids=np.unique(labels[local_seed])
    accepted[weak_ids[sizes[weak_ids]>=5]]=1
    accepted[0]=0
    return accepted[labels]


# Keep all earlier methods. New method is opt-in until visually validated.
_original_detect_ink=detect_ink

def detect_ink(gray, method='legacy'):
    if method=='stroke-core':
        return detect_stroke_cores(gray)
    return _original_detect_ink(gray,method)
