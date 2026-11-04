"""Deferred gutter-shadow geometry carried from spread splitting into cleanup.

Only lightweight, normalized edge-depth samples are stored. No pixels or masks
are changed during splitting. During cleanup, the geometry is transformed using
the same expanded-canvas deskew as the scan, clipped to the final crop, then
text-line rectangles are protected before the shadow pixels are whitened.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def trace_gutter_shadow(gray, split_x, side, n_samples=56, max_depth_fraction=.16):
    """Trace a dark gutter band inward from the chosen split on one half.

    Uses band-averaged, smoothed grayscale profiles to avoid confusing
    individual letters with page-long scanner shading. Return normalized
    [vertical_position, depth] samples or None if no continuous dark band.
    """
    h,w=gray.shape
    half_w=split_x if side=='left' else w-split_x
    if half_w < 40:return None
    max_depth=min(round(max_depth_fraction*half_w), round(.085*w))
    if max_depth < 6:return None
    strip=gray[:, max(0,split_x-max_depth):split_x][:,::-1] if side=='left' else gray[:,split_x:min(w,split_x+max_depth)]
    stripsz=strip.shape[1]
    centers=np.linspace(0,h-1,n_samples)
    depths=np.full(n_samples,np.nan,dtype=np.float64)
    supported=np.zeros(n_samples,bool)
    for j,cy in enumerate(centers):
        radius=max(3,round(h*.009))
        ya=max(0,round(cy)-radius)
        yb=min(h,round(cy)+radius+1)
        if yb<=ya:continue
        profile=strip[ya:yb].astype(np.float32).mean(axis=0)
        profile=cv2.GaussianBlur(profile[None],(0,0),sigmaX=1.2)[0]
        # A gutter is visibly gray close to the cut; blank paper is not.
        if np.median(profile[:min(7,stripsz)]) >= 239:continue
        # Find first sustained clean paper beyond the shadow, not the next
        # dark feature (text). Require several consecutive bright columns.
        clean=(profile >= 244).astype(np.uint8)
        consecutive=np.convolve(clean,np.ones(5,dtype=int),mode='valid')
        indices=np.flatnonzero(consecutive>=5)
        if len(indices):
            depth=max(2,int(indices[0])+3)
        else:
            # Skip unsafe profiles that remain dark all the way into page.
            continue
        if depth>=stripsz-5:continue
        depths[j]=depth
        supported[j]=True
    interior=(centers >= h*.10)&(centers<=h*.90)
    if np.count_nonzero(supported&interior)<max(7,round(np.count_nonzero(interior)*.45)):
        return None
    good=np.flatnonzero(supported)
    yy=np.arange(n_samples)
    depths=np.interp(yy,good,depths[good])
    # Remove high-frequency glitches while retaining slanted trajectories.
    depths=cv2.GaussianBlur(depths[None].astype(np.float32),(0,0),sigmaX=1.3)[0]
    depths=np.clip(depths+1,2,max_depth)
    # Do not save giant masks; these knots scale to any output resolution.
    return [[round(float(y)/(h-1),6),round(float(d)/half_w,6)] for y,d in zip(centers,depths)]


def shadow_mask(shape, samples, side, angle, input_shape, crop_rect, padding=0):
    """Rasterize original split-page shadow coordinates after bound-preserving deskew.

    The polygon is mapped by the same affine used by `rotate_bound`.
    Cropping is applied only after transformation to avoid shifted masks.
    """
    out_h,out_w=shape[:2]
    mask=np.zeros((out_h,out_w),np.uint8)
    if not samples:return mask
    h,w=input_shape[:2]
    cx,cy=w/2,h/2
    M=cv2.getRotationMatrix2D((cx,cy),float(angle),1.0)
    co,si=abs(M[0,0]),abs(M[0,1])
    nw,nh=math.ceil(h*si+w*co),math.ceil(h*co+w*si)
    M[0,2]+=nw/2-cx;M[1,2]+=nh/2-cy
    edge=float(w if side=='right' else 0)
    ys=np.asarray([min(h-1,max(0,s[0]*(h-1))) for s in samples],np.float32)
    depth=np.asarray([max(0,s[1]*w) for s in samples],np.float32)
    xs=edge-depth if side=='right' else edge+depth
    edgepoints=np.column_stack((np.full(len(ys),edge,np.float32),ys))
    innerpoints=np.column_stack((xs,ys))
    polygon=np.vstack((edgepoints,innerpoints[::-1]))
    homogeneous=np.column_stack((polygon,np.ones(len(polygon),dtype=np.float32)))
    mapped=homogeneous@M.T
    x0,y0,_,_=crop_rect
    mapped[:,0]-=x0; mapped[:,1]-=y0
    cv2.fillPoly(mask,[np.rint(mapped).astype(np.int32)],255)
    if padding:
        mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*padding+1,2*padding+1)))
    return mask


def apply_deferred_gutter(rgb, record, angle, input_shape, crop_rect,
                          text_lines=(), safety_pixels=8, return_mask=False):
    """Whiten gutter geometry, excluding all trusted text-line rectangles.

    Parameters
    ----------
    text_lines : iterable
        Rectangles in *deskewed full-page* coordinates; they are translated
        into crop coordinates before protecting pixels. A margin is added.
    """
    if not record or not record.get('samples'):
        zero=np.zeros(rgb.shape[:2],np.uint8)
        return (rgb.copy(),zero) if return_mask else rgb.copy()
    crop_rect=tuple(crop_rect)
    mask=shadow_mask(rgb.shape,record['samples'],record['edge'],angle,
                     input_shape,crop_rect)
    x0,y0,_,_=crop_rect
    sh,sw=mask.shape
    for a,b,c,d,*_ in text_lines:
        xa=max(0,int(math.floor(a-x0))-safety_pixels)
        ya=max(0,int(math.floor(b-y0))-safety_pixels)
        xb=min(sw,int(math.ceil(c-x0))+safety_pixels)
        yb=min(sh,int(math.ceil(d-y0))+safety_pixels)
        if xb>xa and yb>ya:
            mask[ya:yb,xa:xb]=0
    result=rgb.copy()
    result[mask!=0]=255
    return (result,mask) if return_mask else result
