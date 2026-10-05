"""Reproduce the current-source clarity-margin and photometric-guard scan."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from scripts.validate_clear_fusion import make_cases
from xiyuan_mvp.benchmark import LPIPSEvaluator, masked_metrics
from xiyuan_mvp.clear_fusion import fuse

ROOT = Path(__file__).resolve().parents[1]


def main(output: str = "outputs/clear-photometric-v15.json"):
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    cases = []
    for names in (("page", "retina", "cell", "clock"),
                  ("text", "coins", "hubble_deep_field", "moon")):
        cases.extend(make_cases(list(names), ["local_warp", "light_gradient", "blur_noise", "noise"], 640))
    profiles = {
        f"margin_{margin:.2f}": {
            "mode": "adaptive_detail", "detail_scale": 9, "guard": True,
            "clarity_margin": margin, "noise_guard": True,
        }
        for margin in (1.05, 1.10, 1.20, 1.30)
    }
    evaluator = LPIPSEvaluator()
    records = []
    started = time.perf_counter()
    for info, registration, reference in cases:
        valid = np.full(reference.shape[:2], 255, np.uint8)
        valid[:3] = valid[-3:] = 0
        valid[:, :3] = valid[:, -3:] = 0
        row = {"id": info["id"],
               "group": "fresh" if info["source"] in {"page", "retina", "cell", "clock"} else "heldout",
               "scores": {}, "metrics": {}}
        for name, config in profiles.items():
            image, metrics = fuse(registration, config)
            row["scores"][name] = {
                **masked_metrics(image, reference, valid),
                "lpips": evaluator.evaluate(image, reference, valid),
            }
            row["metrics"][name] = metrics
        records.append(row)
    payload = {
        "algorithm_sha256": hashlib.sha256((ROOT / "xiyuan_mvp/clear_fusion.py").read_bytes()).hexdigest(),
        "profiles": profiles, "records": records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target = ROOT / output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for group in ("fresh", "heldout", "all"):
        for name in profiles:
            values = [r["scores"][name] for r in records if group == "all" or r["group"] == group]
            print(name, group, len(values),
                  "mae", round(statistics.mean(v["mae"] for v in values), 6),
                  "lpips", round(statistics.mean(v["lpips"] for v in values), 6),
                  "ssim", round(statistics.mean(v["ssim"] for v in values), 6))


if __name__ == "__main__":
    main()
