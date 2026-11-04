"""Experimental spatial filtering of candidate ink (without rectangle filling).

The returned masks contain only originally detected ink pixels. Grouping is used
solely to decide which pixels to KEEP, never to fill line or paragraph boxes.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def glyph_candidates(binary):
    """Find plausible character components while rejecting long scan bands."""
    binary = (binary > 0).astype(np.uint8)
    n, lab, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    h,w=binary.shape
    chosen=[]
    for k in range(1,n):
        x,y,cw,ch,area=map(int,stats[k])
        if not (2<=cw<=min(48,.055*w) and 3<=ch<=min(43,.045*h)):
            continue
        if not (4<=area<=600 and .065<=area/(cw*ch)<=.90):
            continue
        if not (.12<=cw/ch<=4):
            continue
        chosen.append(k)
    return lab,stats,centers,np.asarray(chosen,dtype=int)


def neighbor_consistency(gray, binary, min_neighbors=2):
    """Keep characters with similarly sized neighbors on nearby baselines.

    No dilation or bounding-box filling occurs. A character must participate in
    a locally horizontal group, which suppresses unrelated vertical streaks.
    """
    lab, stats, centers, ids = glyph_candidates(binary)
    output=np.zeros_like(binary,np.uint8)
    if len(ids)==0:return output
    sizes=stats[ids]
    heights=sizes[:,3].astype(float)
    typical=np.median(heights)
    cx=centers[ids,0];cy=centers[ids,1]
    widths=sizes[:,2]
    # Limit candidate comparisons to neighboring x positions using sorted order.
    order=np.argsort(cx)
    accepted=np.zeros(len(ids),bool)
    for pos in range(len(ids)):
        i=order[pos]
        near=order[max(0,pos-38):min(len(order),pos+39)]
        dx=np.abs(cx[near]-cx[i])
        dy=np.abs(cy[near]-cy[i])
        similar=(heights[near]>=heights[i]*.42)&(heights[near]<=heights[i]*2.3)
        aligned=dy<=max(3.5,.50*typical,.72*heights[i])
        spacing=(dx>max(1,.25*widths[i]))&(dx<max(16,3.8*typical))
        side_left=np.any(similar&aligned&spacing&(cx[near]<cx[i]))
        side_right=np.any(similar&aligned&spacing&(cx[near]>cx[i]))
        count=np.count_nonzero(similar&aligned&spacing)
        accepted[i]=count>=min_neighbors and (side_left or side_right)
    lookup=np.zeros(int(lab.max())+1,np.uint8)
    lookup[ids[accepted]]=1
    return lookup[lab]


def baseline_projection(gray,binary):
    """Keep source ink along repeatedly populated horizontal text baselines.

    Uses component geometry, horizontal row projection and connected runs;
    results still contain only original candidate-ink pixels.
    """
    lab, stats, centers, ids=glyph_candidates(binary)
    result=np.zeros_like(binary,np.uint8)
    if len(ids)==0:return result
    h,w=binary.shape
    candidates=np.zeros(len(ids),bool)
    yc=centers[ids,1]
    # Sweep groups of plausible components in a narrow baseline neighborhood.
    heights=stats[ids,3].astype(float)
    typ=max(5,float(np.median(heights)))
    ordered=np.argsort(yc)
    for idx in ordered:
        window=np.abs(yc-yc[idx]) < max(4.5,.60*typ)
        nx=len(np.unique((centers[ids[window],0]//max(3,typ*.55)).astype(int)))
        extent=np.ptp(centers[ids[window],0]) if window.sum()>1 else 0
        candidates[idx]=(window.sum()>=6 and nx>=5 and extent>max(24,3.8*typ))
    lookup=np.zeros(int(lab.max())+1,np.uint8)
    lookup[ids[candidates]]=1
    return lookup[lab]


def union_filter(gray,binary):
    """Conservative union of two independent character-structure filters."""
    return np.maximum(neighbor_consistency(gray,binary),baseline_projection(gray,binary))


def tuned_baseline(gray, binary, min_components=4, min_columns=3,
                   horizontal_extent=2.4, vertical_tolerance=0.90):
    """Retain original ink belonging to a horizontally populated character row.

    Compared with :func:`baseline_projection`, this uses a more permissive
    row-height neighborhood and requires fewer components and x-columns. It
    does *not* fill rectangles or enlarge the retained character pixels.

    Parameters
    ----------
    gray : ndarray
        Original grayscale image (reserved for future intensity checks).
    binary : ndarray
        Strict raw ink candidate mask.
    min_components : int
        Minimum candidate characters in a local horizontal group.
    min_columns : int
        Number of horizontally distinct character positions required.
    horizontal_extent : float
        Minimum group span in units of median glyph height.
    vertical_tolerance : float
        Allowed baseline deviation, in units of median glyph height.

    Returns
    -------
    ndarray
        uint8 mask retaining original foreground pixels only.
    """
    lab,stats,centers,ids=glyph_candidates(binary)
    if not len(ids):return np.zeros_like(binary,np.uint8)
    heights=stats[ids,3].astype(float)
    typ=max(4.,float(np.median(heights)))
    cy=centers[ids,1]; cx=centers[ids,0]
    keep=np.zeros(len(ids),bool)
    for i in range(len(ids)):
        near=np.abs(cy-cy[i])<=max(4.5, vertical_tolerance*typ)
        xs=cx[near]
        buckets=np.unique((xs/max(3.,typ*.50)).astype(int))
        keep[i]=(len(xs)>=min_components and len(buckets)>=min_columns
                 and np.ptp(xs)>=max(14.,horizontal_extent*typ))
    lookup=np.zeros(int(lab.max())+1,np.uint8)
    lookup[ids[keep]]=1
    return lookup[lab]


def tuned_horizontal_neighbors(gray,binary,min_neighbors=1,
                               max_horizontal_glyphs=5.5,
                               vertical_tolerance=0.72):
    """Filter original glyph pixels by nearby *horizontal* neighbors.

    The test is intentionally permissive: one nearby character is sufficient,
    but the gap must be primarily horizontal rather than vertical. Character
    size differences are allowed to accommodate headings and punctuation.
    """
    lab,stats,centers,ids=glyph_candidates(binary)
    if not len(ids):return np.zeros_like(binary,np.uint8)
    heights=stats[ids,3].astype(float)
    typ=max(4.,float(np.median(heights)))
    cx=centers[ids,0];cy=centers[ids,1]
    keep=np.zeros(len(ids),bool)
    for i in range(len(ids)):
        dx=np.abs(cx-cx[i]);dy=np.abs(cy-cy[i])
        similar=(heights>=.32*heights[i])&(heights<=3.3*heights[i])
        horizontal=(dx>max(1.5,.17*typ))&(dx<max(18.,max_horizontal_glyphs*typ))
        aligned=(dy<=max(4.,vertical_tolerance*typ))&(dx>=1.4*dy)
        keep[i]=np.count_nonzero(similar&horizontal&aligned)>=min_neighbors
    lookup=np.zeros(int(lab.max())+1,np.uint8)
    lookup[ids[keep]]=1
    return lookup[lab]


def height_based_horizontal_neighbors(gray, binary, min_neighbors=2,
                                      max_gap_heights=2.0,
                                      vertical_tolerance=0.65,
                                      height_ratio=2.5, split_wide=True):
    """Retain horizontally supported groups of character-like ink components.

    Examine larger candidate components first. A seed qualifies when it has
    ``min_neighbors`` distinct plausible neighbors, OR at least one plausible
    neighbor already accepted into a qualifying text group. When it qualifies,
    *all* plausible neighbors within its search window are accepted together.
    This deliberately accepts word-end components that would not independently
    pass the two-neighbor rule.

    Parameters
    ----------
    gray : numpy.ndarray
        Source grayscale image, currently reserved for compatibility.
    binary : numpy.ndarray
        Candidate foreground (nonzero pixels represent ink).
    min_neighbors : int, optional
        Required plausible neighbors for an unconnected group seed (default 2).
    max_gap_heights : float, optional
        Maximum edge-to-edge horizontal gap in glyph heights (default 2).
    vertical_tolerance : float, optional
        Maximum vertical centroid distance in local glyph heights (default 0.65).
    height_ratio : float, optional
        Maximum ratio of neighboring component heights (default 2.5).

    Returns
    -------
    numpy.ndarray
        Uint8 0/1 pixel mask consisting only of accepted original components.

    Notes
    -----
    When ``split_wide`` is true, wide components are logically split at
    vertical ink-density valleys *before* the neighbor test. All source ink
    pixels remain intact; erosion is not used.

    Acceptance is monotone: previously accepted components are never removed.
    Accepted components are skipped as qualification seeds. Each accepted
    component is expanded at most once to discover the rest of its group;
    neighborhood lookups are cached. This prevents repeated seed testing.
    """
    from bisect import bisect_left, bisect_right
    from collections import deque

    binary = (binary > 0).astype(np.uint8)
    if split_wide:
        from .component_splitting import split_wide_components
        labels, stats, centers, _ = split_wide_components(binary)
        n = len(stats)
    else:
        n, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    h, w = binary.shape
    chosen = []
    for k in range(1, n):
        x, y, cw, ch, area = map(int, stats[k])
        if not (3 <= ch <= min(48, .055*h) and 2 <= cw <= min(180, .18*w)):
            continue
        if not (4 <= area <= 2500 and .04 <= area/max(1,cw*ch) <= .95):
            continue
        chosen.append(k)
    if not chosen:
        return np.zeros_like(binary)

    ids = np.asarray(chosen, dtype=np.int32)
    comp = stats[ids]
    left = comp[:, 0].astype(float)
    right = left + comp[:, 2]
    heights = comp[:, 3].astype(float)
    areas = comp[:, cv2.CC_STAT_AREA]
    cy = centers[ids, 1]
    cx = centers[ids, 0]
    order = np.argsort(left)
    left_sorted = left[order]
    typical = max(4., float(np.median(heights)))
    keep = np.zeros(len(ids), dtype=bool)
    cache = {}

    def neighbors_of(i):
        """All geometrically compatible components in the seed's window."""
        if i in cache:
            return cache[i]
        reach = max_gap_heights * max(typical, heights[i])
        stop = bisect_right(left_sorted, right[i] + reach)
        start = bisect_left(left_sorted, left[i] - reach - min(180., .18*w))
        possible = order[start:stop]
        if possible.size == 0:
            cache[i] = np.empty(0, np.int32)
            return cache[i]
        gap = np.maximum(0, np.maximum(left[possible] - right[i],
                                       left[i] - right[possible]))
        hlocal = np.maximum(heights[possible], heights[i])
        same_height = ((heights[possible] >= heights[i]/height_ratio) &
                       (heights[possible] <= heights[i]*height_ratio))
        aligned = np.abs(cy[possible] - cy[i]) <= np.maximum(3., vertical_tolerance*hlocal)
        separated = np.abs(cx[possible] - cx[i]) >= np.maximum(
            2., .25*np.minimum(heights[possible], heights[i]))
        nearby = gap <= max_gap_heights*hlocal
        result = possible[same_height & aligned & separated & nearby & (possible != i)]
        cache[i] = result
        return result

    # Separate qualification from expansion. Accepted components are never
    # re-tested as seeds. Their neighbor sets are explored at most once during
    # the expansion stage, even when several seeds discover the same group.
    seed_order = np.argsort(-areas, kind='stable')
    expanded = np.zeros(len(ids), dtype=bool)
    from collections import deque

    def grow_group(start):
        """Accept and traverse an entire geometrically connected text group.

        Each component's neighborhood is inspected only once, and all
        compatible components are accepted together. This avoids repeating
        two-neighbor tests on previously accepted letters.
        """
        pending = deque(map(int, np.atleast_1d(start)))
        while pending:
            j = pending.popleft()
            keep[j] = True
            if expanded[j]:
                continue
            expanded[j] = True
            nearby = neighbors_of(j)
            new = nearby[~keep[nearby]]
            if new.size:
                keep[new] = True
                pending.extend(map(int, new))

    for i in map(int, seed_order):
        if keep[i]:
            continue  # Already accepted by another group; no redundant test.
        matches = neighbors_of(i)
        # A qualifying seed must have two neighbors, or one already-accepted
        # neighbor. The latter check also handles two-component group ends.
        if len(matches) >= min_neighbors or (len(matches) and np.any(keep[matches])):
            grow_group(np.r_[i, matches])

    lookup = np.zeros(n, np.uint8)
    lookup[ids[keep]] = 1
    return lookup[labels]


def height_neighbors_with_wide_glyph_rescue(gray, binary, **kwargs):
    """Two-neighbor height filter with conservative rescue for touching glyphs.

    Wide connected glyph groups may contain several joined letters and thus
    have fewer than two *separate* neighbors. Rescue only wide, character-height
    components independently supported by the tuned horizontal baseline filter.
    Long vertical streaks remain excluded by glyph candidate dimensions.
    """
    primary = height_based_horizontal_neighbors(gray, binary, **kwargs)
    baseline = tuned_baseline(gray, binary)
    lab, stats, _, ids = glyph_candidates(binary)
    rescued = np.zeros(int(lab.max())+1, np.uint8)
    for k in ids:
        x,y,w,h,area=map(int,stats[k])
        if w >= 1.35*h and np.any((lab[y:y+h,x:x+w]==k) & (baseline[y:y+h,x:x+w]>0)):
            rescued[k] = 1
    return np.maximum(primary, rescued[lab])
