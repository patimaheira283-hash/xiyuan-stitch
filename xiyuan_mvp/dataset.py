from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import cv2
import numpy as np

from .image_io import write_image
from .run_io import sha256, write_json


def create_dataset(output: Path) -> Path:
    from skimage import data
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("数据集已存在，请使用新目录。")
    sources = {
        "brick": (data.brick, "https://cc0textures.com/view.php?tex=Bricks25"),
        "grass": (data.grass, "https://www.deviantart.com/linolafett/art/Grass-01-434853879"),
        "gravel": (data.gravel, "https://cc0textures.com/view.php?tex=Gravel04"),
    }
    variants = ["identity", "brighter", "darker", "gamma", "perspective", "rotate",
                "noise", "resolution", "lighting_gradient", "mixed"]
    cases, source_records = [], []
    for category, (loader, url) in sources.items():
        reference = cv2.cvtColor(loader(), cv2.COLOR_GRAY2BGR)
        original = output / "sources" / f"{category}.png"
        write_image(original, reference)
        source_records.append({
            "id": category, "file": str(original.relative_to(output)).replace("\\", "/"),
            "sha256": sha256(original), "source_url": url, "license": "CC0-1.0",
            "packaged_by": "scikit-image", "loader_documentation": inspect.getdoc(loader),
        })
        for index, variant in enumerate(variants):
            case_id = f"{category}_{index + 1:02d}_{variant}"
            folder = output / "cases" / case_id
            a = reference[:, :340].copy()
            b = reference[:, 172:].copy()
            rng = np.random.default_rng(2026 + index)
            if variant in ("brighter", "mixed"):
                b = np.clip(b.astype(float) * 1.15 + 12, 0, 255).astype(np.uint8)
            elif variant == "darker":
                b = np.clip(b.astype(float) * 0.80 - 6, 0, 255).astype(np.uint8)
            elif variant == "gamma":
                b = np.clip((b / 255.0)**1.3 * 255, 0, 255).astype(np.uint8)
            elif variant == "noise":
                b = np.clip(b.astype(float) + rng.normal(0, 5, b.shape), 0, 255).astype(np.uint8)
            elif variant == "resolution":
                b = cv2.resize(cv2.resize(b, (170, 256)), (340, 512))
            elif variant == "lighting_gradient":
                gain = np.linspace(0.75, 1.2, b.shape[0])[:, None, None]
                b = np.clip(b.astype(float) * gain, 0, 255).astype(np.uint8)
            if variant in ("perspective", "mixed"):
                h = np.array([[1, 0.012, 2], [0.006, 1, 2], [0.000015, 0.000020, 1]], dtype=float)
                b = cv2.warpPerspective(b, h, (340, 512), borderMode=cv2.BORDER_REFLECT_101)
            elif variant == "rotate":
                h = cv2.getRotationMatrix2D((170, 256), 1.2, 1)
                b = cv2.warpAffine(b, h, (340, 512), borderMode=cv2.BORDER_REFLECT_101)
            write_image(folder / "a.png", a)
            write_image(folder / "b.png", b)
            relative = lambda p: str(p.relative_to(output)).replace("\\", "/")
            cases.append({
                "id": case_id, "category": category, "source_id": category,
                "kind": "derived_photo_pair", "variant": variant,
                "image_a": relative(folder / "a.png"), "image_b": relative(folder / "b.png"),
                "reference": relative(original), "reference_to_a": np.eye(3).tolist(),
                "input_sha256": {"a": sha256(folder / "a.png"), "b": sha256(folder / "b.png")},
            })
    write_json(output / "sources.json", {"sources": source_records})
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "description": "30 controlled pairs derived from 3 CC0 grayscale photographs; not 30 independent real captures.",
        "independent_source_count": 3, "case_count": len(cases), "cases": cases,
    })
    (output / "README.md").write_text(
        "# 可控纹理测试集\n\n"
        "30 组输入来自 3 张独立 CC0 照片（砖墙、草地、碎石），每张生成 10 种扰动。"
        "这不是 30 组独立实拍数据，也不包含木纹。用途是调试、回归测试和有参考图的受控实验。\n\n"
        "reference 为原始照片，reference_to_a 将它映射到图片 A 坐标。"
        "真实场景结论还需补充独立拍摄的图片，不能以这些派生样本夸大样本量。"
        "来源、许可证及 scikit-image 函数原文见 sources.json。\n", encoding="utf-8")
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description="准备 CC0 照片派生测试集")
    parser.add_argument("--output", default="data/benchmark")
    args = parser.parse_args()
    print(create_dataset(Path(args.output)).resolve())


if __name__ == "__main__":
    main()
