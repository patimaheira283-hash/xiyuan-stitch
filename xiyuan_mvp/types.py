from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class RegistrationResult:
    image_a: np.ndarray
    image_b: np.ndarray
    canvas_a: np.ndarray
    canvas_b: np.ndarray
    mask_a: np.ndarray
    mask_b: np.ndarray
    overlap_mask: np.ndarray
    homography_b_to_a: np.ndarray
    canvas_transform: np.ndarray
    match_image: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaskResult:
    binary_mask: np.ndarray
    soft_mask: np.ndarray
    bbox: tuple[int, int, int, int]
    method: str = "center"


@dataclass(slots=True)
class StitchResult:
    traditional_image: np.ndarray
    final_image: np.ndarray
    registration: RegistrationResult
    mask: MaskResult
    metrics: dict[str, Any] = field(default_factory=dict)
    poisson_image: np.ndarray | None = None
    repair_region: np.ndarray | None = None
    source_images: list[np.ndarray] = field(default_factory=list)
    repair_base_image: np.ndarray | None = None
