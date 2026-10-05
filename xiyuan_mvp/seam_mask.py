from __future__ import annotations

import cv2
import numpy as np

from .errors import StitchError
from .types import MaskResult, RegistrationResult


def _center_seam(overlap: np.ndarray) -> np.ndarray:
    ys, xs = np.where(overlap > 0)
    if len(xs) == 0:
        raise StitchError("两张图片没有有效重叠区域。")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    line = np.zeros_like(overlap)

    if (x1 - x0) <= (y1 - y0):
        for y in range(y0, y1 + 1):
            row_x = np.where(overlap[y] > 0)[0]
            if row_x.size:
                x = int((row_x[0] + row_x[-1]) / 2)
                line[y, x] = 255
    else:
        for x in range(x0, x1 + 1):
            column_y = np.where(overlap[:, x] > 0)[0]
            if column_y.size:
                y = int((column_y[0] + column_y[-1]) / 2)
                line[y, x] = 255
    return line


def generate_seam_mask(
    registration: RegistrationResult, config: dict
) -> MaskResult:
    seam_width = max(3, int(config.get("seam_width", 64)))
    method = str(config.get("method", "center"))
    line = _minimum_cost_seam(registration) if method == "min_cost" else None
    actual_method = "min_cost" if line is not None else "center"
    if line is None:
        line = _center_seam(registration.overlap_mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (seam_width | 1, seam_width | 1)
    )
    binary = cv2.dilate(line, kernel)
    binary = cv2.bitwise_and(binary, registration.overlap_mask)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    binary = cv2.bitwise_and(binary, registration.overlap_mask)
    feather = max(1, int(config.get("feather_radius", 24)))
    kernel_size = feather * 2 + 1
    soft = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)
    soft = cv2.bitwise_and(soft, registration.overlap_mask)

    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        raise StitchError("无法生成有效接缝 Mask。")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    return MaskResult(binary_mask=binary, soft_mask=soft, bbox=bbox, method=actual_method)


def mask_from_binary(binary, allowed, feather_radius=24, method="custom"):
    binary = cv2.bitwise_and(np.where(binary > 127, 255, 0).astype(np.uint8), allowed)
    ys, xs = np.where(binary > 0)
    if not xs.size:
        raise StitchError("没有有效的待修复区域。")
    radius = int(feather_radius)
    soft = cv2.GaussianBlur(binary, (2*radius+1, 2*radius+1), 0)
    soft = cv2.bitwise_and(soft, allowed)
    return MaskResult(binary, soft, (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)), method)


def _minimum_cost_seam(registration):
    overlap = registration.overlap_mask
    x, y, width, height = cv2.boundingRect(overlap)
    if min(width, height) < 3:
        return None
    a = registration.canvas_a[y:y+height, x:x+width]
    b = registration.canvas_b[y:y+height, x:x+width]
    valid = overlap[y:y+height, x:x+width] > 0
    cost = np.mean(np.abs(a.astype(np.float32)-b.astype(np.float32)), axis=2)/255
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for dx, dy in ((1, 0), (0, 1)):
        cost += 0.08*np.abs(cv2.Sobel(ga,cv2.CV_32F,dx,dy)-cv2.Sobel(gb,cv2.CV_32F,dx,dy))/255
    transpose = width > height
    if transpose:
        cost, valid = cost.T, valid.T
    h, w = cost.shape
    # Bound the DP memory for very large exports; return to original coordinates afterwards.
    scale = min(1.0, 900/max(h,w))
    size = (max(3,int(w*scale)), max(3,int(h*scale)))
    small = cv2.resize(cost,size,interpolation=cv2.INTER_AREA)
    usable = cv2.resize(valid.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    sh, sw = small.shape
    small += (np.abs(np.arange(sw)-(sw-1)/2)/max(sw,1)*0.02)[None,:]
    small[~usable] = 1e9
    previous = small[0].copy()
    directions = np.zeros((sh,sw),np.int8)
    for row in range(1,sh):
        choices=np.stack((np.r_[1e9,previous[:-1]],previous,np.r_[previous[1:],1e9]))
        directions[row]=choices.argmin(axis=0).astype(np.int8)-1
        previous=small[row]+choices.min(axis=0)
    end=int(previous.argmin())
    if previous[end]>=1e9:
        return None
    points=[]
    for row in range(sh-1,-1,-1):
        points.append((min(w-1,int((end+0.5)*w/sw)),min(h-1,int((row+0.5)*h/sh))))
        end+=int(directions[row,end])
    line=np.zeros((h,w),np.uint8)
    cv2.polylines(line,[np.asarray(points,np.int32)],False,255,1)
    if transpose:
        line=line.T
    output=np.zeros_like(overlap)
    output[y:y+height,x:x+width]=line
    return cv2.bitwise_and(output,overlap)
