from __future__ import annotations

import cv2
import numpy as np

from xiyuan_mvp.config import load_config
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.inpainting import extract_patch, restore_patch


def _synthetic_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    base = rng.integers(0, 40, size=(520, 1200, 3), dtype=np.uint8)
    for x in range(30, 1170, 60):
        cv2.line(base, (x, 10), (x, 510), (100 + x % 120, 180, 220), 2)
    for y in range(30, 500, 55):
        cv2.line(base, (10, y), (1190, y), (220, 100 + y % 120, 160), 2)
    for index in range(24):
        center = (50 + index * 45, 80 + (index * 83) % 360)
        cv2.circle(base, center, 8 + index % 7, (230, 230, 230), -1)
    cv2.putText(base, "XIYUAN MVP", (410, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    first = base[:, :760].copy()
    second = base[:, 440:].copy()
    second = cv2.convertScaleAbs(second, alpha=1.03, beta=4)
    return first, second


def test_traditional_pipeline_smoke() -> None:
    first, second = _synthetic_pair()
    config = load_config()
    result = StitchPipeline(config).run(first, second)
    assert result.final_image.shape[0] >= 500
    assert result.final_image.shape[1] >= 1150
    assert result.metrics["good_matches"] >= config["registration"]["min_matches"]
    assert result.metrics["inlier_ratio"] >= config["registration"]["min_inlier_ratio"]
    assert np.count_nonzero(result.mask.binary_mask) > 0
    assert np.array_equal(result.final_image, result.traditional_image)


def test_patch_round_trip_preserves_aspect_ratio() -> None:
    image = np.zeros((200, 600, 3), dtype=np.uint8)
    image[:, :, 1] = np.arange(600, dtype=np.uint8)[None, :]
    mask = np.zeros((200, 600), dtype=np.uint8)
    mask[80:120, 260:340] = 255
    patch, patch_mask, transform = extract_patch(
        image, mask, (260, 80, 340, 120), context_pixels=80, patch_size=512
    )
    restored = restore_patch(patch, transform)
    assert patch.shape == (512, 512, 3)
    assert patch_mask.shape == (512, 512)
    assert restored.shape[:2] == (transform.original_height, transform.original_width)
    inner_width = transform.inner_x1 - transform.inner_x0
    inner_height = transform.inner_y1 - transform.inner_y0
    assert abs(inner_width / inner_height - transform.original_width / transform.original_height) < 0.02


def test_ai_branch_with_fake_inpainter() -> None:
    class FakeInpainter:
        @staticmethod
        def generate(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
            result = image_bgr.copy()
            result[mask > 0] = (40, 200, 240)
            return result

    first, second = _synthetic_pair()
    pipeline = StitchPipeline(load_config())
    pipeline._inpainter = FakeInpainter()  # type: ignore[assignment]
    result = pipeline.run(first, second, use_ai=True)
    assert result.metrics["use_ai"] is True
    assert result.metrics["ai_seconds"] >= 0
    assert not np.array_equal(result.final_image, result.traditional_image)
    outside = result.mask.soft_mask == 0
    assert np.array_equal(result.final_image[outside], result.traditional_image[outside])
