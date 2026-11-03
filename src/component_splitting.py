"""Split wide ink components at very deep vertical valleys, without removing ink.

A cut is accepted only if each side retains ink spanning at least half of the
original component height. Input pixels are unchanged; the return value is a
new logical labeling used only for recognizing text components.
"""
import cv2
import numpy as np


def split_wide_components(binary, min_width_ratio=1.0,
                          valley_fraction=0.15, min_side_height_fraction=0.50,
                          min_fragment_width_ratio=0.30,
                          max_cuts_per_component=16):
    """Assign separate logical IDs to wide connected components at deep valleys.

    Parameters
    ----------
    binary : numpy.ndarray
        uint8 ink mask; nonzero denotes candidate ink.
    min_width_ratio : float, optional
        Consider splitting only when component width >= this times height.
    valley_fraction : float, optional
        Column ink count at the cut divided by the maximum *local* column
        count must be <= this fraction. Default is 0.15.
    min_side_height_fraction : float, optional
        The vertical extent of ink on *both* sides of the cut must each be
        >= this proportion of the full component height. Default is 0.50.
    min_fragment_width_ratio : float, optional
        Reject pieces narrower than this many component heights, preventing
        a word from being sliced into narrow strokes.
    max_cuts_per_component : int, optional
        Safety cap on the number of cuts in any component.

    Returns
    -------
    labels, stats, centers, details : tuple
        Logical labels of the original ink pixels, OpenCV-format component
        statistics, centroid array, and diagnostic counts/cut coordinates.

    Notes
    -----
    The true vertical extent (max row - min row) of nonzero pixels on each
    side is measured; column ink count is used only for valley depth. The
    original pixels are not eroded, deleted, or overwritten.
    """
    ink = (binary > 0).astype(np.uint8)
    n, original_labels, source_stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    page_h, page_w = ink.shape
    heights = source_stats[1:, cv2.CC_STAT_HEIGHT]
    widths = source_stats[1:, cv2.CC_STAT_WIDTH]
    plausible = heights[(heights >= 4) & (heights <= min(45, .06*page_h)) & (widths <= .22*page_w)]
    typical = float(np.median(plausible)) if len(plausible) else 12.0

    labels = np.zeros_like(original_labels, dtype=np.int32)
    next_id = 0
    stats_rows = [np.zeros(5, dtype=np.int32)]
    centers_rows = [np.zeros(2, dtype=np.float64)]
    cuts = []
    considered = 0
    for k in range(1, n):
        x, y, w, h, area = map(int, source_stats[k])
        component = (original_labels[y:y+h, x:x+w] == k)
        selected = []
        if (4 <= h <= min(48, .07*page_h) and
                w >= max(9, min_width_ratio*h) and w <= .25*page_w):
            considered += 1
            counts = np.count_nonzero(component, axis=0)
            # A narrow column may have zero ink, in which case the original
            # object would not be connected, so deep *nonzero* valleys matter.
            # Compare to nearby peaks rather than unrelated spikes elsewhere.
            local_radius = max(3, h)
            margin = max(2, int(round(min_fragment_width_ratio*h)))
            candidates = []
            for j in range(margin, w-margin):
                if counts[j] > counts[j-1] or counts[j] > counts[j+1]:
                    continue
                lp = int(np.max(counts[max(0,j-local_radius):j]))
                rp = int(np.max(counts[j+1:min(w,j+local_radius+1)]))
                peak = min(lp, rp)
                if peak < max(3, .30*h) or counts[j] > valley_fraction*peak:
                    continue
                # Check actual ink's vertical span on each side, not the
                # column's ink count (which can miss thin but tall glyphs).
                left_rows = np.flatnonzero(np.any(component[:, :j], axis=1))
                right_rows = np.flatnonzero(np.any(component[:, j:], axis=1))
                if len(left_rows) == 0 or len(right_rows) == 0:
                    continue
                left_height = int(left_rows[-1]-left_rows[0]+1)
                right_height = int(right_rows[-1]-right_rows[0]+1)
                if min(left_height, right_height) < min_side_height_fraction*h:
                    continue
                candidates.append((counts[j]/max(peak, 1), -min(left_height,right_height), j))
            # Select deepest cuts first, and require viable fragment width.
            for _, _, j in sorted(candidates):
                trial = sorted(selected + [j])
                if min(np.diff([0]+trial+[w])) < margin:
                    continue
                selected.append(j)
                if len(selected) >= max_cuts_per_component:
                    break
        boundaries = [0] + sorted(selected) + [w]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            ys, xs = np.where(component[:, lo:hi])
            if not len(xs):
                continue
            next_id += 1
            roi = labels[y:y+h, x+lo:x+hi]
            roi[ys, xs] = next_id
            x_abs = x+lo+int(xs.min())
            y_abs = y+int(ys.min())
            stats_rows.append(np.asarray((x_abs,y_abs,int(xs.max()-xs.min()+1),
                                          int(ys.max()-ys.min()+1),len(xs)), np.int32))
            centers_rows.append(np.asarray((x+lo+float(xs.mean()),y+float(ys.mean())), np.float64))
        cuts.extend((x+j,y,y+h) for j in sorted(selected))
    stats = np.stack(stats_rows)
    centers = np.stack(centers_rows)
    details = {'original_components':n-1, 'logical_components':next_id,
               'cuts':cuts,'typical_height':typical, 'considered':considered}
    return labels,stats,centers,details
