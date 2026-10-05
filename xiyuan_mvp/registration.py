from __future__ import annotations

import cv2
import numpy as np

from .errors import RegistrationError
from .types import RegistrationResult


def _resize_for_features(image: np.ndarray, max_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    resized = image if scale == 1.0 else cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    transform = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return resized, transform


def _detect_matches(
    image_a: np.ndarray, image_b: np.ndarray, ratio_test: float
) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch], str]:
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=8000)
        norm = cv2.NORM_L2
        method = "SIFT"
    else:
        detector = cv2.ORB_create(nfeatures=8000)
        norm = cv2.NORM_HAMMING
        method = "ORB"

    keypoints_a, descriptors_a = detector.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(gray_b, None)
    if descriptors_a is None or descriptors_b is None:
        raise RegistrationError("图片纹理不足，无法提取有效特征。")

    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio_test * pair[1].distance]
    good.sort(key=lambda match: match.distance)
    return keypoints_a, keypoints_b, good, method


def _build_canvas(
    image_a: np.ndarray, image_b: np.ndarray, homography: np.ndarray
) -> tuple[np.ndarray, tuple[int, int]]:
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    corners_a = np.float32(
        [[0, 0], [width_a, 0], [width_a, height_a], [0, height_a]]
    ).reshape(-1, 1, 2)
    corners_b = np.float32(
        [[0, 0], [width_b, 0], [width_b, height_b], [0, height_b]]
    ).reshape(-1, 1, 2)
    warped_b = cv2.perspectiveTransform(corners_b, homography)
    all_corners = np.concatenate([corners_a, warped_b], axis=0).reshape(-1, 2)
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    translation = np.array(
        [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    size = (int(max_x - min_x), int(max_y - min_y))
    if size[0] <= 0 or size[1] <= 0 or size[0] * size[1] > 150_000_000:
        raise RegistrationError("估计的画布尺寸异常，配准结果不可用。")
    return translation, size


def _register_images(
    image_a: np.ndarray, image_b: np.ndarray, config: dict,
    valid_mask_a: np.ndarray | None = None, valid_mask_b: np.ndarray | None = None,
) -> RegistrationResult:
    cv2.setRNGSeed(int(config.get("seed", 2026)) % (2**31 - 1))
    max_size = int(config.get("max_feature_size", 1600))
    small_a, scale_a = _resize_for_features(image_a, max_size)
    small_b, scale_b = _resize_for_features(image_b, max_size)
    if config.get("method", "sift") == "loftr":
        from .loftr import detect_matches
        keypoints_a, keypoints_b, matches, method = detect_matches(small_a, small_b, config)
    else:
        keypoints_a, keypoints_b, matches, method = _detect_matches(
            small_a, small_b, float(config.get("ratio_test", 0.75))
        )
    for mask, scale, points, is_a in ((valid_mask_a, scale_a, keypoints_a, True),
                                      (valid_mask_b, scale_b, keypoints_b, False)):
        if mask is not None:
            matches = [m for m in matches if mask[
                min(mask.shape[0]-1, max(0, int(points[m.trainIdx if is_a else m.queryIdx].pt[1]/scale[1, 1]))),
                min(mask.shape[1]-1, max(0, int(points[m.trainIdx if is_a else m.queryIdx].pt[0]/scale[0, 0])))
            ] > 0]

    min_matches = int(config.get("min_matches", 12))
    if len(matches) < min_matches:
        raise RegistrationError(
            f"有效匹配点只有 {len(matches)} 个，至少需要 {min_matches} 个。"
        )

    source = np.float32([keypoints_b[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([keypoints_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    h_small, inlier_mask = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        float(config.get("ransac_threshold", 4.0)),
    )
    if h_small is None or inlier_mask is None:
        raise RegistrationError("RANSAC 无法估计有效的单应矩阵。")

    inliers = int(inlier_mask.ravel().sum())
    inlier_ratio = inliers / len(matches)
    if inliers < int(config.get("min_inliers", 8)):
        raise RegistrationError(f"RANSAC 内点只有 {inliers} 个，配准不可靠。")
    if inlier_ratio < float(config.get("min_inlier_ratio", 0.25)):
        raise RegistrationError(f"RANSAC 内点比例仅为 {inlier_ratio:.1%}，配准不可靠。")

    homography = np.linalg.inv(scale_a) @ h_small @ scale_b
    if not np.isfinite(homography).all() or abs(homography[2, 2]) < 1e-10:
        raise RegistrationError("单应矩阵异常，无法构建画布。")
    homography /= homography[2, 2]
    translation, canvas_size = _build_canvas(image_a, image_b, homography)
    canvas_a = cv2.warpPerspective(image_a, translation, canvas_size)
    canvas_b = cv2.warpPerspective(image_b, translation @ homography, canvas_size)
    source_mask_a = np.full(image_a.shape[:2], 255, dtype=np.uint8) if valid_mask_a is None else valid_mask_a
    source_mask_b = np.full(image_b.shape[:2], 255, dtype=np.uint8) if valid_mask_b is None else valid_mask_b
    mask_a = cv2.warpPerspective(source_mask_a, translation, canvas_size)
    mask_b = cv2.warpPerspective(source_mask_b, translation @ homography, canvas_size)
    mask_a = np.where(mask_a > 127, 255, 0).astype(np.uint8)
    mask_b = np.where(mask_b > 127, 255, 0).astype(np.uint8)
    overlap = cv2.bitwise_and(mask_a, mask_b)
    overlap_pixels = int(np.count_nonzero(overlap))
    smaller_area = min(int(np.count_nonzero(mask_a)), int(np.count_nonzero(mask_b)))
    overlap_ratio = overlap_pixels / max(smaller_area, 1)
    if overlap_ratio < float(config.get("min_overlap_ratio", 0.10)):
        raise RegistrationError(f"变换后的有效重叠区域仅为 {overlap_ratio:.1%}。")

    draw_mask = inlier_mask.ravel().astype(np.uint8).tolist()
    match_image = cv2.drawMatches(
        small_b,
        keypoints_b,
        small_a,
        keypoints_a,
        matches,
        None,
        matchesMask=draw_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    metrics = {
        "feature_method": method,
        "raw_keypoints_a": len(keypoints_a),
        "raw_keypoints_b": len(keypoints_b),
        "good_matches": len(matches),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "overlap_ratio": overlap_ratio,
        "canvas_width": canvas_size[0],
        "canvas_height": canvas_size[1],
    }
    return RegistrationResult(
        image_a=image_a,
        image_b=image_b,
        canvas_a=canvas_a,
        canvas_b=canvas_b,
        mask_a=mask_a,
        mask_b=mask_b,
        overlap_mask=overlap,
        homography_b_to_a=homography,
        canvas_transform=translation,
        match_image=match_image,
        metrics=metrics,
    )


def register_images(image_a, image_b, config, valid_mask_a=None, valid_mask_b=None):
    method = config.get("method", "sift")
    if method not in ("sift", "loftr", "auto"):
        raise RegistrationError("配准方法必须是 sift、loftr 或 auto。")
    for image, mask in ((image_a, valid_mask_a), (image_b, valid_mask_b)):
        if mask is not None and mask.shape != image.shape[:2]:
            raise RegistrationError("有效图像区域与输入尺寸不一致。")
    if method != "auto":
        return _register_images(image_a, image_b, config, valid_mask_a, valid_mask_b)
    try:
        return _register_images(image_a, image_b, {**config, "method": "sift"}, valid_mask_a, valid_mask_b)
    except RegistrationError as first_error:
        result = _register_images(image_a, image_b, {**config, "method": "loftr"}, valid_mask_a, valid_mask_b)
        result.metrics["sift_fallback_reason"] = str(first_error)
        return result
