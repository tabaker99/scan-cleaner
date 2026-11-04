"""Group neighbor-approved ink into horizontal text lines (experimental method promoted).

This module only groups already accepted ink; it performs no further character
rejection. Compatible groups are merged with union-find across gaps up to 3 em.
"""
from bisect import bisect_left, bisect_right
import cv2
import numpy as np

def build_grouped_lines(accepted, max_gap_ems=3.0, vertical_tolerance=0.48,
                        height_ratio=2.5, minimum_pixels=1):
    """Form horizontal text lines by merging accepted ink components.

    Parameters
    ----------
    accepted : ndarray
        uint8 foreground pixels already approved by a text filter; no further
        connected-component shape rejection is performed.
    max_gap_ems : float
        Maximum edge-to-edge search distance in units of *typical character
        height*, default 3.0.
    vertical_tolerance : float
        Maximum allowable vertical center disagreement as a fraction of the
        larger character height.
    height_ratio : float
        Maximum component height ratio to connect by an edge.
    minimum_pixels : int
        Skip isolated tiny blobs only when set above 1; default keeps all.

    Returns
    -------
    accepted : ndarray
        Unaltered approved character pixels.
    rectangles : list[tuple]
        (left, top, right, bottom, component_count) for all merged groups.
    membership : dict[int, list[int]]
        Component labels belonging to each group; each approved component
        appears in exactly one group.
    """
    pixels = (accepted > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(pixels, 8)
    ids = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= minimum_pixels) + 1
    if len(ids) == 0:
        return pixels, [], {}
    x0 = stats[ids, cv2.CC_STAT_LEFT].astype(np.float64)
    y0 = stats[ids, cv2.CC_STAT_TOP].astype(np.float64)
    widths = stats[ids, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[ids, cv2.CC_STAT_HEIGHT].astype(np.float64)
    x1, y1 = x0 + widths, y0 + heights
    cx, cy = centroids[ids, 0], centroids[ids, 1]
    # Robust typical height; avoid punctuation fragments lowering the unit.
    typical_height = max(3.0, float(np.median(heights[heights >= np.percentile(heights, 40)])))
    limit = float(max_gap_ems) * typical_height
    order = np.argsort(x0)
    sorted_x = x0[order]
    parent = np.arange(len(ids))
    size = np.ones(len(ids), dtype=np.int32)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    edges = 0
    for i in range(len(ids)):
        # Search horizontal continuation in *both* directions. This includes
        # all compatible components within 3 ems, not only two neighbors.
        lo = bisect_left(sorted_x, x0[i] - limit - np.max(widths))
        hi = bisect_right(sorted_x, x1[i] + limit)
        candidate = order[lo:hi]
        candidate = candidate[candidate > i]
        if not len(candidate):
            continue
        gap = np.maximum(0, np.maximum(x0[candidate] - x1[i], x0[i] - x1[candidate]))
        shared_h = np.maximum(heights[i], heights[candidate])
        heights_ok = ((heights[candidate] >= heights[i]/height_ratio) &
                      (heights[candidate] <= heights[i]*height_ratio))
        y_ok = np.abs(cy[candidate] - cy[i]) <= np.maximum(2.5, vertical_tolerance*shared_h)
        # A strong vertical overlap guard avoids merging consecutive baselines.
        overlap = np.maximum(0, np.minimum(y1[candidate], y1[i]) - np.maximum(y0[candidate], y0[i]))
        overlap_ok = overlap >= 0.30 * np.minimum(heights[i], heights[candidate])
        horizontal = abs(cx[candidate]-cx[i]) >= .28*np.minimum(heights[i],heights[candidate])
        good = candidate[(gap <= limit) & heights_ok & y_ok & overlap_ok & horizontal]
        for j in good:
            union(i, int(j))
        edges += len(good)

    members = {}
    for i, label in enumerate(ids):
        members.setdefault(root(i), []).append(i)
    rectangles = []
    membership = {}
    for key, indices in members.items():
        indices = np.array(indices)
        rectangles.append((int(np.min(x0[indices])), int(np.min(y0[indices])),
                           int(np.max(x1[indices])), int(np.max(y1[indices])),len(indices)))
        membership[int(key)] = [int(ids[i]) for i in indices]
    rectangles.sort(key=lambda r: (r[1], r[0]))
    return pixels, rectangles, membership

