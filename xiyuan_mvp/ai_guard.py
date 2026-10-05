"""Input-only safety checks for generated seam repairs."""
from __future__ import annotations

import cv2
import numpy as np


def boundary_discontinuity(image: np.ndarray, binary_mask: np.ndarray, radius: int = 3) -> float:
    """Measure high-frequency energy around the editable boundary.

    This is a runtime plausibility signal, not a perceptual quality score. It
    uses only the produced image and the user mask, so it cannot leak a target
    or reference image into repair.
    """
    if image.ndim != 3 or binary_mask.ndim != 2 or image.shape[:2] != binary_mask.shape:
        raise ValueError("边界保护的图像和 Mask 尺寸不一致。")
    mask = (binary_mask > 0).astype(np.uint8)
    if not np.any(mask):
        return 0.0
    radius = max(1, int(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    band = (cv2.dilate(mask, kernel) > 0) & (cv2.erode(mask, kernel) == 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), max(1.0, radius / 1.5)))
    return float(detail[band].mean()) if np.any(band) else 0.0


def apply_ai_guard(base: np.ndarray, candidate: np.ndarray, binary_mask: np.ndarray,
                   soft_mask: np.ndarray, blend_config: dict) -> tuple[np.ndarray, dict]:
    """Scale an already color-matched, feathered candidate toward its base.

    Both pipeline layouts apply the soft mask in blend_generated_patch before
    calling this function. Here the mask limits support only; multiplying it
    again would square the feather, even with the guard disabled.
    """
    if base.shape != candidate.shape or base.shape[:2] != binary_mask.shape or soft_mask.shape != binary_mask.shape:
        raise ValueError("AI 边界保护的输入尺寸不一致。")
    requested = float(np.clip(blend_config.get("ai_opacity", 1.0), 0.0, 1.0))
    guard = bool(blend_config.get("ai_boundary_guard", False))
    base_score = boundary_discontinuity(base, binary_mask) if guard else None
    candidate_score = boundary_discontinuity(candidate, binary_mask) if guard else None
    threshold = max(1.0, float(blend_config.get("ai_boundary_threshold", 1.15)))
    effective = requested
    triggered = False
    allowed_score = max(base_score * threshold, base_score + 0.25) if guard else None
    if guard and candidate_score > allowed_score:
        # Keep the candidate's useful signal when the regression is moderate,
        # but move continuously toward the protected structural base.
        effective *= float(np.clip(allowed_score / max(candidate_score, 1e-6), 0.0, 1.0))
        triggered = True
    output = np.clip(np.rint(base.astype(np.float64) * (1.0 - effective)
                            + candidate.astype(np.float64) * effective), 0, 255).astype(np.uint8)
    output[soft_mask == 0] = base[soft_mask == 0]
    return output, {
        "ai_blend_revision": "single-feather-v2",
        "ai_opacity": requested,
        "ai_boundary_guard": guard,
        "ai_boundary_guard_triggered": triggered,
        "ai_boundary_base_score": base_score,
        "ai_boundary_candidate_score": candidate_score,
        "ai_requested_opacity": requested,
        "ai_effective_opacity": effective,
    }
