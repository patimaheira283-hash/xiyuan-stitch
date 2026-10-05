from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import cv2
import numpy as np

from .errors import RegistrationError


@lru_cache(maxsize=2)
def _model(pretrained: str, device: str, weights: str):
    try:
        import torch
        from kornia.feature import LoFTR
        from kornia.feature.loftr.loftr import urls
    except ImportError as exc:
        raise RegistrationError('LoFTR 依赖未安装，请安装 .[loftr] 可选依赖。') from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RegistrationError("LoFTR 指定 CUDA，但没有可用 NVIDIA GPU。")
    try:
        model = LoFTR(pretrained=None)
        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        else:
            if pretrained not in urls:
                raise ValueError(f"未知 LoFTR 预训练模型：{pretrained}")
            state = torch.hub.load_state_dict_from_url(
                urls[pretrained].replace("http://", "https://"), map_location="cpu", weights_only=True,
            )
        model.load_state_dict(state.get("state_dict", state))
        return model.eval().to(device)
    except Exception as exc:
        raise RegistrationError(f"LoFTR 权重加载失败：{exc}") from exc


def detect_matches(image_a: np.ndarray, image_b: np.ndarray, config: dict):
    try:
        import torch
    except ImportError as exc:
        raise RegistrationError("LoFTR 需要安装 PyTorch 与 Kornia。") from exc
    device = str(config.get("loftr_device", "cpu"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    max_size = int(config.get("loftr_max_size", 640))
    def tensor(image):
        h, w = image.shape[:2]
        scale = min(1.0, max_size / max(h, w))
        size = (max(8, int(w*scale)//8*8), max(8, int(h*scale)//8*8))
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size)
        return torch.from_numpy(gray.copy()).to(device=device, dtype=torch.float32)[None, None]/255, np.array([w/size[0], h/size[1]])
    a, scale_a = tensor(image_a)
    b, scale_b = tensor(image_b)
    model = _model(str(config.get("loftr_pretrained", "outdoor")), device, str(config.get("loftr_weights", "")))
    with torch.inference_mode():
        output = model({"image0": b, "image1": a})
    confidence = output["confidence"].cpu().numpy()
    keep = np.flatnonzero(confidence >= float(config.get("loftr_confidence", 0.2)))
    keep = keep[np.argsort(-confidence[keep])][:8000]
    source = output["keypoints0"].cpu().numpy()[keep] * scale_b
    target = output["keypoints1"].cpu().numpy()[keep] * scale_a
    kp_a = [cv2.KeyPoint(float(x), float(y), 1) for x, y in target]
    kp_b = [cv2.KeyPoint(float(x), float(y), 1) for x, y in source]
    matches = [cv2.DMatch(i, i, float(1-confidence[k])) for i, k in enumerate(keep)]
    return kp_a, kp_b, matches, "LoFTR"
