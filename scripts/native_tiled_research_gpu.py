"""Compare legacy resized diffusion patches with native-scale tiled patches."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
import traceback
import zipfile

import torch

from scripts.research_gpu import inputs, OUTPUT, DATA
from scripts.deformation_cases import create_cases
from scripts.refined_research_gpu import quality
from xiyuan_mvp.benchmark import LPIPSEvaluator, make_comparison
from xiyuan_mvp.config import load_config
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import environment, save_failure, save_run, write_json


PROFILES = {
    "legacy_canny": {"controlnet": "canny", "patch_layout": "single"},
    "native_canny": {"controlnet": "canny", "patch_layout": "native_tiled"},
    "legacy_tile": {"controlnet": "tile", "patch_layout": "single"},
    "native_tile": {"controlnet": "tile", "patch_layout": "native_tiled"},
}


def main() -> None:
    started = time.perf_counter()
    records = []
    report = {"status": "running", "experiment": "native-scale tiled diffusion vs resized single patch",
              "protocol": {"opacity": 0.2, "same seed and steps": True,
                           "references_used_by_repair": False}}
    try:
        assert torch.cuda.is_available()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1, 8*1024**3/total))
        report.update(gpu=torch.cuda.get_device_name(0), actual_vram_bytes=total,
                      allocator_limit_bytes=min(total, 8*1024**3), environment=environment())
        config = load_config()
        config["registration"].update(method="loftr", loftr_device="cuda", loftr_max_size=640)
        config["refinement"]["enabled"] = True
        config["blend"]["ai_opacity"] = 0.2
        config["inpainting"].update(device="cuda", local_files_only=False, variant="fp16",
            cache_dir="/kaggle/temp/xiyuan-hf-cache", steps=20, strength=.25,
            guidance_scale=3.5, controlnet_scale=.8, patch_size=512, tile_overlap=128)
        cases = inputs() + create_cases(DATA/"deformation")
        write_json(OUTPUT/"inputs-tiled-v6.json", {"cases": cases, "profiles": PROFILES, "config": config})
        evaluator = LPIPSEvaluator(); prepared = {}
        for case in cases:
            try:
                baseline = StitchPipeline(config).run(case["image_a"], case["image_b"])
                prepared[case["id"]] = baseline
                records.append({"id": case["id"], "profile": "structural_only", "status": "success",
                    "quality": quality({"Structural": baseline.repair_base_image}, case, baseline, evaluator),
                    "metrics": baseline.metrics})
            except Exception as exc:
                records.append({"id": case["id"], "profile": "structural_only", "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}"})
            write_json(OUTPUT/"progress-tiled-v6.json", {"records": records})
        pipe = StitchPipeline(deepcopy(config))
        for profile, options in PROFILES.items():
            pipe.config["inpainting"].update(options)
            for case in cases:
                row = {"id": case["id"], "profile": profile, "status": "failed"}
                folder = OUTPUT/profile/case["id"]
                try:
                    if case["id"] not in prepared: raise ValueError("registration failed")
                    result = pipe.run(case["image_a"], case["image_b"], prepared=prepared[case["id"]], use_ai=True)
                    outside = result.mask.soft_mask == 0
                    base = prepared[case["id"]].repair_base_image
                    assert (result.final_image[outside] == base[outside]).all()
                    row.update(status="success", metrics=result.metrics,
                               quality=quality({profile: result.final_image}, case, result, evaluator),
                               outside_mask_unchanged=True)
                    save_run(folder, result, pipe.config, extra={"case": case, "profile": profile})
                    make_comparison(folder/"comparison.png", {"Structural": base, profile: result.final_image}, result.mask.bbox)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    save_failure(folder, exc, pipe.config)
                    traceback.print_exc()
                records.append(row)
                write_json(folder/"evaluation.json", row)
                write_json(OUTPUT/"progress-tiled-v6.json", {"records": records})
        report["status"] = "completed_with_failures" if any(r["status"] != "success" for r in records) else "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(records=records, elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT/"report-tiled-v6.json", report)
        with zipfile.ZipFile("/kaggle/working/native-tiled-v6-bundle.zip", "w", zipfile.ZIP_STORED) as archive:
            for path in OUTPUT.rglob("*"):
                if path.is_file(): archive.write(path, path.relative_to(OUTPUT.parent))


if __name__ == "__main__": main()
