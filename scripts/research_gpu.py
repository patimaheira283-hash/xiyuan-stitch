from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import requests
import torch

from xiyuan_mvp.benchmark import LPIPSEvaluator, make_comparison, masked_metrics
from xiyuan_mvp.config import load_config
from xiyuan_mvp.dataset import create_dataset
from xiyuan_mvp.image_io import read_image, write_image
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import environment, save_run, save_failure, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/kaggle/working/research-results")
DATA = Path("/kaggle/temp/xiyuan-research-data")

PROFILES = {
    "plain": {"controlnet": "none", "use_lcm": False, "steps": 20, "guidance_scale": 4.0, "strength": 0.35},
    "canny": {"controlnet": "canny", "use_lcm": False, "steps": 20, "guidance_scale": 4.0, "strength": 0.35},
    "tile": {"controlnet": "tile", "use_lcm": False, "steps": 20, "guidance_scale": 4.0, "strength": 0.35},
    "canny_lcm": {"controlnet": "canny", "use_lcm": True, "steps": 8, "guidance_scale": 1.0, "strength": 0.5},
    "tile_lcm": {"controlnet": "tile", "use_lcm": True, "steps": 8, "guidance_scale": 1.0, "strength": 0.5},
}


def inputs():
    controlled = DATA / "controlled"
    manifest_path = create_dataset(controlled)
    controlled_cases = json.loads(manifest_path.read_text())["cases"]
    # Development subset only; held-out cases remain available for the final benchmark.
    selected = [c for c in controlled_cases if c["variant"] in ("brighter", "mixed")]
    cases = []
    for c in selected:
        cases.append({**c, "image_a": str(controlled/c["image_a"]), "image_b": str(controlled/c["image_b"]),
                      "reference": str(controlled/c["reference"]), "split": "development"})
    real = json.loads((ROOT / "data/research-cases.json").read_text(encoding="utf-8"))["cases"]
    # One or two scenes per repository, chosen before observing algorithm results.
    count = {}
    for c in real:
        count[c["category"]] = count.get(c["category"], 0)+1
        if count[c["category"]] > 2:
            continue
        folder = DATA / "real" / c["id"]
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, item in enumerate(c["files"]):
            response = requests.get(item["url"], timeout=90)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != item["sha256"]:
                raise ValueError(f"Source checksum mismatch: {c['id']}")
            raw_path = folder / f"original_{i}.img"
            raw_path.write_bytes(response.content)
            image = read_image(raw_path)
            h, w = image.shape[:2]
            scale = min(1, 1200/max(h, w))
            if scale < 1:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            path = folder/f"input_{i}.png"
            write_image(path, image)
            paths.append(str(path))
        cases.append({**c, "image_a": paths[0], "image_b": paths[1], "split": "development",
                      "resizing": "long side capped at 1200 for development; original bytes and hashes verified"})
    write_json(OUTPUT / "inputs.json", {"cases": cases})
    return cases


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    report = {"status": "running", "experiment": "development ablation, not final held-out result"}
    try:
        assert torch.cuda.is_available(), "CUDA unavailable"
        assert (torch.ones(4, device="cuda")*2).sum().item() == 8
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 8*1024**3/total), 0)
        report.update(gpu=torch.cuda.get_device_name(0), actual_vram_bytes=total,
                      allocator_limit_bytes=min(total, 8*1024**3),
                      budget_note="8 GiB PyTorch allocator cap on T4; not a native 8GB device test.",
                      environment=environment())
        config = load_config()
        config["registration"].update(method="loftr", loftr_device="cuda", loftr_max_size=640)
        config["inpainting"].update(
            device="cuda", local_files_only=False, variant="fp16",
            cache_dir="/kaggle/temp/xiyuan-hf-cache",
        )
        cases = inputs()
        evaluator = LPIPSEvaluator()
        pipe = StitchPipeline(config)
        prepared = {}
        for case in cases:
            record = {"id": case["id"], "stage": "registration", "status": "failed", "methods": {}}
            for method in ("sift", "loftr"):
                try:
                    local_config = deepcopy(config)
                    local_config["registration"]["method"] = method
                    baseline = StitchPipeline(local_config).run(case["image_a"], case["image_b"])
                    record["methods"][method] = {"status": "success", **baseline.metrics}
                    save_run(OUTPUT / "registration" / case["id"] / method, baseline, local_config, extra={"case": case})
                    if method == "loftr":
                        prepared[case["id"]] = baseline
                        record["status"] = "success"
                except Exception as exc:
                    record["methods"][method] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            write_json(OUTPUT / "progress.json", {"records": records})
            print("REGISTRATION", json.dumps(record), flush=True)
        # Mode-major order reuses model weights within each profile.
        for profile, options in PROFILES.items():
            pipe.config["inpainting"].update(options)
            print("PROFILE", profile, flush=True)
            for case in cases:
                if case["id"] not in prepared:
                    continue
                folder = OUTPUT / profile / case["id"]
                record = {"id": case["id"], "profile": profile, "status": "failed", "split": case["split"]}
                try:
                    baseline = prepared[case["id"]]
                    result = pipe.run(case["image_a"], case["image_b"], prepared=baseline,
                                      use_ai=True, progress=lambda msg,n: print(n,msg,flush=True))
                    torch.cuda.synchronize()
                    if result.metrics.get("test_double"):
                        raise AssertionError("Fake inference is not allowed")
                    outside = result.mask.soft_mask == 0
                    difference = np.abs(result.final_image.astype(np.int16)-baseline.traditional_image.astype(np.int16))
                    assert not np.any(difference[outside]), "Pixels outside the repair mask changed"
                    record.update(status="success", metrics=result.metrics,
                                  outside_mask_max_difference=int(difference[outside].max()) if outside.any() else 0,
                                  inside_mask_mean_difference=float(difference[result.mask.binary_mask > 0].mean()))
                    save_run(folder, result, pipe.config, extra={"case": case, "profile": profile})
                    images = {"Feather": baseline.traditional_image, "Poisson": baseline.poisson_image, profile: result.final_image}
                    if case.get("reference"):
                        reference = read_image(case["reference"])
                        reg = result.registration
                        h,w = result.final_image.shape[:2]
                        transform = reg.canvas_transform @ np.asarray(case["reference_to_a"])
                        ref = cv2.warpPerspective(reference, transform, (w,h))
                        valid = cv2.warpPerspective(np.full(reference.shape[:2],255,np.uint8),transform,(w,h))
                        valid = cv2.erode(valid,np.ones((7,7),np.uint8))
                        mask = cv2.bitwise_and(valid,result.mask.binary_mask)
                        quality = {}
                        for name,image in images.items():
                            quality[name] = {**masked_metrics(image,ref,mask),
                                             "lpips_alex_spatial": evaluator.evaluate(image,ref,mask)}
                        record["quality"] = quality
                        images = {"Reference":ref,**images}
                    else:
                        record["quality"] = None
                        record["quality_note"] = "No reference panorama exists; use a blind visual review."
                    make_comparison(folder/"comparison.png",images,result.mask.bbox)
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    save_failure(folder,exc,pipe.config,extra={"case":case,"profile":profile})
                    traceback.print_exc()
                records.append(record)
                write_json(folder/"evaluation.json",record)
                write_json(OUTPUT/"progress.json",{"records":records})
                print("RESULT",json.dumps(record),flush=True)
        report["status"] = "completed_with_failures" if any(r["status"] != "success" for r in records) else "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        report.update(records=records,elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT/"report.json",report)
        print("FINAL",json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
