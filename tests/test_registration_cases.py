from __future__ import annotations

import cv2
import numpy as np
import pytest

from xiyuan_mvp.config import load_config
from xiyuan_mvp.errors import RegistrationError
from xiyuan_mvp.registration import register_images


def _feature_scene(height: int = 720, width: int = 960) -> np.ndarray:
    rng = np.random.default_rng(708)
    image = rng.integers(0, 55, size=(height, width, 3), dtype=np.uint8)
    for index in range(80):
        x = int(rng.integers(20, width - 20))
        y = int(rng.integers(20, height - 20))
        radius = int(rng.integers(4, 15))
        color = tuple(int(value) for value in rng.integers(90, 255, size=3))
        cv2.circle(image, (x, y), radius, color, -1)
    for y in range(40, height, 80):
        cv2.line(image, (0, y), (width - 1, y + 15), (200, 170, 230), 2)
    cv2.putText(
        image,
        "XIYUAN",
        (width // 4, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (250, 250, 250),
        5,
    )
    return image


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)


def test_vertical_translation_registration() -> None:
    scene = _feature_scene(1000, 700)
    image_a = scene[:680].copy()
    image_b = scene[320:].copy()
    result = register_images(image_a, image_b, load_config()["registration"])
    assert result.metrics["inlier_ratio"] > 0.60
    assert abs(result.homography_b_to_a[1, 2] - 320) < 2.0
    assert result.overlap_mask.any()


def test_mild_perspective_registration_accuracy() -> None:
    image_a = _feature_scene()
    height, width = image_a.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([[18, 10], [width - 25, 25], [width - 8, height - 20], [28, height - 5]])
    a_to_b = cv2.getPerspectiveTransform(source, target)
    image_b = cv2.warpPerspective(image_a, a_to_b, (width, height))
    result = register_images(image_a, image_b, load_config()["registration"])
    expected = np.linalg.inv(a_to_b)
    probe = np.float32([[100, 100], [width - 100, 100], [width - 100, height - 100], [100, height - 100]])
    error = np.linalg.norm(_project(result.homography_b_to_a, probe) - _project(expected, probe), axis=1)
    assert float(error.mean()) < 2.0
    assert result.metrics["inlier_ratio"] > 0.50


def test_low_texture_failure_is_clear() -> None:
    image_a = np.full((480, 640, 3), 120, dtype=np.uint8)
    image_b = np.full((480, 640, 3), 130, dtype=np.uint8)
    with pytest.raises(RegistrationError, match="纹理不足"):
        register_images(image_a, image_b, load_config()["registration"])

