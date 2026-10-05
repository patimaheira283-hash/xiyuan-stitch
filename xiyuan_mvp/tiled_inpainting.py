"""Native-scale diffusion patches read from a fixed base and blended once."""
from __future__ import annotations

import cv2
import numpy as np

from .blending import blend_generated_patch
from .ai_guard import apply_ai_guard
from .errors import CancelledError


def _starts(low: int, high: int, size: int, overlap: int) -> list[int]:
    if high - low <= size:
        return [low]
    starts = list(range(low, high - size + 1, size - overlap))
    if starts[-1] != high - size:
        starts.append(high - size)
    return starts


def plan_tiles(binary: np.ndarray, soft: np.ndarray, size: int, overlap: int, context: int):
    if binary.shape != soft.shape or binary.ndim != 2:
        raise ValueError("分块 Mask 必须为尺寸一致的二维数组。")
    if size < 8 or not 0 < overlap < size or context < 0:
        raise ValueError("分块参数无效：重叠宽度必须小于 Patch 尺寸。")
    region = cv2.bitwise_or(binary, soft)
    x, y, w, h = cv2.boundingRect(region)
    if not w or not h:
        raise ValueError("分块修复区域为空。")
    height, width = binary.shape
    x0, y0 = max(0, x-context), max(0, y-context)
    x1, y1 = min(width, x+w+context), min(height, y+h+context)
    tiles = []
    for top in _starts(y0, y1, size, overlap):
        for left in _starts(x0, x1, size, overlap):
            right, bottom = min(left+size, x1), min(top+size, y1)
            if np.any(region[top:bottom, left:right]):
                tiles.append((left, top, right, bottom))
    return (x0, y0, x1, y1), tiles


def _weights(box, roi, overlap):
    x0, y0, x1, y1 = box
    width, height = x1-x0, y1-y0
    wx, wy = np.ones(width, np.float32), np.ones(height, np.float32)
    if x0 > roi[0]: wx *= np.minimum((np.arange(width)+1)/overlap, 1)
    if x1 < roi[2]: wx *= np.minimum((width-np.arange(width))/overlap, 1)
    if y0 > roi[1]: wy *= np.minimum((np.arange(height)+1)/overlap, 1)
    if y1 < roi[3]: wy *= np.minimum((height-np.arange(height))/overlap, 1)
    return wy[:, None]*wx[None, :]


def repair_tiled(base, mask, ai_config, blend_config, generate, *, progress=None, cancelled=None):
    """generate(patch, mask, callback) returns (BGR prediction, metrics).

    Reflection padding is removed by slicing, never resizing. Every tile sees
    the same immutable base. Weighted predictions are combined before color
    matching and the user's opacity are applied, so overlaps are not repainted
    repeatedly. Nothing is returned if a tile fails or the user cancels.
    """
    size = int(ai_config["patch_size"])
    overlap = int(ai_config.get("tile_overlap", 128))
    roi, tiles = plan_tiles(mask.binary_mask, mask.soft_mask, size, overlap,
                            int(ai_config.get("context_pixels", 192)))
    x0, y0, x1, y1 = roi
    total = np.zeros((y1-y0, x1-x0, 3), np.float32)
    weights = np.zeros((y1-y0, x1-x0), np.float32)
    records = []
    for index, box in enumerate(tiles):
        if cancelled and cancelled():
            raise CancelledError("已取消分块生成，保留底图。")
        left, top, right, bottom = box
        crop = base[top:bottom, left:right]
        binary = mask.binary_mask[top:bottom, left:right]
        height, width = binary.shape
        pad_x, pad_y = (size-width)//2, (size-height)//2
        padded = cv2.copyMakeBorder(crop, pad_y, size-height-pad_y, pad_x,
                                    size-width-pad_x, cv2.BORDER_REFLECT_101)
        padded_mask = cv2.copyMakeBorder(binary, pad_y, size-height-pad_y, pad_x,
                                         size-width-pad_x, cv2.BORDER_CONSTANT, value=0)
        def notify(message, percent):
            if progress:
                progress(f"分块 {index+1}/{len(tiles)} · {message}",
                         70 + int(25*(index + np.clip(percent, 0, 100)/100)/len(tiles)))
        if np.any(binary):
            prediction, metrics = generate(padded, padded_mask, notify)
            if prediction.shape != padded.shape or prediction.dtype != np.uint8:
                raise ValueError("生成分块的尺寸或像素类型与输入不一致。")
            prediction = prediction[pad_y:pad_y+height, pad_x:pad_x+width]
        else:
            # A feather-only edge tile needs no denoising.
            prediction, metrics = crop, {"inference_skipped": "empty_binary_mask"}
        if cancelled and cancelled():
            raise CancelledError("已取消分块生成，保留底图。")
        weight = _weights(box, roi, overlap)
        target = np.s_[top-y0:bottom-y0, left-x0:right-x0]
        total[target] += prediction.astype(np.float32)*weight[:, :, None]
        weights[target] += weight
        records.append({"box": list(box), **metrics})
    support = mask.soft_mask[y0:y1, x0:x1] > 0
    if np.any(support & (weights == 0)):
        raise AssertionError("分块计划未覆盖完整修复区域。")
    native = base[y0:y1, x0:x1]
    candidate = native.copy()
    covered = weights > 0
    candidate[covered] = np.clip(np.rint(total[covered]/weights[covered, None]), 0, 255).astype(np.uint8)
    composed = blend_generated_patch(native, candidate, mask.binary_mask[y0:y1, x0:x1],
                                    mask.soft_mask[y0:y1, x0:x1], bool(blend_config.get("color_match", True)), round_output=True)
    candidate = base.copy()
    candidate[y0:y1, x0:x1] = composed
    output, guard_metrics = apply_ai_guard(
        base, candidate, mask.binary_mask, mask.soft_mask, blend_config,
    )
    executed = [row for row in records if not row.get("inference_skipped")]
    metrics = {key: value for key, value in (executed[0] if executed else {}).items()
               if key != "box"}
    metrics.update(patch_layout="native_tiled", patch_scale=1.0, tile_count=len(tiles),
                   generated_tile_count=len(executed), tile_records=records, **guard_metrics)
    for key in ("inference_seconds", "model_load_seconds"):
        metrics[key] = sum(row.get(key, 0.) for row in executed)
    for key in ("peak_vram_mb", "peak_reserved_vram_mb"):
        values = [row[key] for row in executed if row.get(key) is not None]
        metrics[key] = max(values) if values else None
    return output, metrics
