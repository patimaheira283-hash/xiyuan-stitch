from __future__ import annotations

import cv2
import numpy as np

from .types import RegistrationResult


def poisson_blend(registration: RegistrationResult) -> np.ndarray:
    """Clone B's overlap onto an A-first composite; keep exclusive pixels intact."""
    base = registration.canvas_a.copy()
    only_b = (registration.mask_b > 0) & (registration.mask_a == 0)
    base[only_b] = registration.canvas_b[only_b]
    # seamlessClone requires a destination border around its mask.
    mask = registration.overlap_mask.copy()
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0
    x, y, width, height = cv2.boundingRect(mask)
    if width < 3 or height < 3:
        return base
    result = cv2.seamlessClone(
        # OpenCV erodes its mask argument in place. Keep our original support
        # for restoring pixels outside the requested clone region afterwards.
        registration.canvas_b, base, mask.copy(),
        (x + width // 2, y + height // 2), cv2.NORMAL_CLONE,
    )
    result[mask == 0] = base[mask == 0]
    return result


def feather_blend(registration: RegistrationResult, *, float_output: bool = False) -> np.ndarray:
    mask_a = (registration.mask_a > 0).astype(np.uint8)
    mask_b = (registration.mask_b > 0).astype(np.uint8)
    distance_a = cv2.distanceTransform(mask_a, cv2.DIST_L2, 5)
    distance_b = cv2.distanceTransform(mask_b, cv2.DIST_L2, 5)
    denominator = distance_a + distance_b
    weight_a = np.divide(
        distance_a,
        denominator,
        out=np.zeros_like(distance_a),
        where=denominator > 0,
    )
    only_a = (mask_a > 0) & (mask_b == 0)
    only_b = (mask_b > 0) & (mask_a == 0)
    weight_a[only_a] = 1.0
    weight_a[only_b] = 0.0
    weight_a = weight_a[..., None]
    result = (
        registration.canvas_a.astype(np.float32) * weight_a
        + registration.canvas_b.astype(np.float32) * (1.0 - weight_a)
    )
    valid = ((mask_a | mask_b) > 0)[..., None]
    if float_output:
        return np.where(valid,np.clip(result,0,255),0).astype(np.float32)
    return np.where(valid, np.clip(result, 0, 255), 0).astype(np.uint8)


def _color_match(generated: np.ndarray, base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ring_kernel = np.ones((17, 17), dtype=np.uint8)
    outer = cv2.dilate(mask, ring_kernel)
    inner = cv2.erode(mask, ring_kernel)
    ring = (outer > 0) & (inner == 0)
    if np.count_nonzero(ring) < 64:
        return generated

    gen_lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    adjusted = gen_lab.copy()
    for channel in range(3):
        gen_values = gen_lab[..., channel][ring]
        base_values = base_lab[..., channel][ring]
        gen_std = max(float(gen_values.std()), 1.0)
        gain = np.clip(float(base_values.std()) / gen_std, 0.85, 1.15)
        shift = np.clip(float(base_values.mean()) - float(gen_values.mean()) * gain, -18, 18)
        adjusted[..., channel] = adjusted[..., channel] * gain + shift
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)


def blend_generated_patch(
    base: np.ndarray,
    generated: np.ndarray,
    binary_mask: np.ndarray,
    soft_mask: np.ndarray,
    color_match: bool = True,
    *, round_output: bool = False,
) -> np.ndarray:
    corrected = _color_match(generated, base, binary_mask) if color_match else generated
    alpha = (soft_mask.astype(np.float32) / 255.0)[..., None]
    result = base.astype(np.float32) * (1.0 - alpha) + corrected.astype(np.float32) * alpha
    if round_output:
        result = np.rint(result)
    return np.clip(result, 0, 255).astype(np.uint8)
