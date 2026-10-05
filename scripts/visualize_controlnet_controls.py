"""Export the three Canny ControlNet conditioning variants for one prepared run.

This is a preprocessing experiment only; it does not claim that a control
image improves a diffusion result until the GPU comparison job confirms it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from xiyuan_mvp.image_io import read_image, write_image
from xiyuan_mvp.inpainting import control_image, extract_patch
from xiyuan_mvp.pipeline import StitchPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_a", type=Path)
    parser.add_argument("image_b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = StitchPipeline()
    result = pipeline.run(args.image_a, args.image_b)
    patch, patch_mask, transform = extract_patch(
        result.repair_base_image if result.repair_base_image is not None else result.traditional_image,
        result.mask.binary_mask,
        result.mask.bbox,
        int(pipeline.config["inpainting"]["context_pixels"]),
        int(pipeline.config["inpainting"]["patch_size"]),
    )
    write_image(args.output / "00_patch.png", patch)
    write_image(args.output / "01_mask.png", cv2.cvtColor(patch_mask, cv2.COLOR_GRAY2BGR))

    configs = {
        "02_raw_canny.png": {"controlnet_mask_mode": "none"},
        "03_context_edges.png": {"controlnet_mask_mode": "context_edges"},
        "04_faded_edges.png": {
            "controlnet_mask_mode": "faded_edges",
            "controlnet_edge_fade": 8,
            "controlnet_edge_floor": 0.15,
        },
    }
    summary = {}
    for filename, config in configs.items():
        image = np.asarray(control_image(patch, "canny", config, patch_mask))
        write_image(args.output / filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        gray = image[:, :, 0]
        summary[filename] = {
            "all_nonzero": int(np.count_nonzero(gray)),
            "all_intensity_sum": int(gray.sum()),
            "inside_nonzero": int(np.count_nonzero(gray[patch_mask > 0])),
            "inside_intensity_sum": int(gray[patch_mask > 0].sum()),
            "outside_nonzero": int(np.count_nonzero(gray[patch_mask == 0])),
            "outside_intensity_sum": int(gray[patch_mask == 0].sum()),
            "bbox": [transform.x0, transform.y0, transform.x1, transform.y1],
        }
    import json
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
