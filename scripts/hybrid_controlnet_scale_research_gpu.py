"""Tune Canny ControlNet conditioning strength after structural-gain gating."""
from __future__ import annotations

from copy import deepcopy
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
    "gate_canny_scale_040": {"patch_layout": "single", "controlnet": "canny", "controlnet_scale": .4, "blend_opacity": .05},
    "gate_canny_scale_080": {"patch_layout": "single", "controlnet": "canny", "controlnet_scale": .8, "blend_opacity": .05},
    "gate_canny_scale_100": {"patch_layout": "single", "controlnet": "canny", "controlnet_scale": 1.0, "blend_opacity": .05},
}


def main():
    started = time.perf_counter()
    records = []
    report = {"status": "running", "experiment": "hybrid gated ControlNet scale tuning", "protocol": {
        "references_used_by_repair": False, "ai_opacity": .1,
        "diffusion_policy": "on_structural_gain", "min_residual_error": 2.0,
        "profiles": PROFILES,
    }}
    try:
        assert torch.cuda.is_available()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1, 8 * 1024**3 / total))
        report.update(gpu=torch.cuda.get_device_name(0), actual_vram_bytes=total,
                      allocator_limit_bytes=min(total, 8 * 1024**3), environment=environment())
        cfg = load_config()
        cfg["registration"].update(method="loftr", loftr_device="cuda", loftr_max_size=640)
        cfg["clear_fusion"].update(enabled=True, mode="adaptive_detail", guard=True)
        cfg["neural_alignment"].update(device="cuda", method="raft_large", preserve_detail=True)
        cfg["inpainting"].update(engine="hybrid", device="cuda", local_files_only=False,
            variant="fp16", cache_dir="/kaggle/temp/xiyuan-hf-cache", steps=20,
            strength=.25, guidance_scale=3.5, controlnet="canny", controlnet_scale=.8,
            patch_size=512, tile_overlap=128, use_lcm=False)
        cfg["inpainting"].update(hybrid_diffusion_policy="on_structural_gain", hybrid_min_residual_error=2.0)
        cfg["blend"].update(ai_opacity=.1, ai_boundary_guard=True, ai_boundary_threshold=1.15)
        cases = inputs() + create_cases(DATA / "deformation")
        write_json(OUTPUT / "inputs-hybrid-scale-v11.json", {"cases": cases, "profiles": PROFILES, "config": cfg})
        ev = LPIPSEvaluator()
        prepared = {}
        for case in cases:
            try:
                base = StitchPipeline(cfg).run(case["image_a"], case["image_b"])
                prepared[case["id"]] = base
                records.append({"id": case["id"], "profile": "clear_structural_only", "status": "success",
                    "quality": quality({"Structural": base.repair_base_image}, case, base, ev), "metrics": base.metrics})
            except Exception as exc:
                records.append({"id": case["id"], "profile": "clear_structural_only", "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"})
            write_json(OUTPUT / "progress-hybrid-scale-v11.json", {"records": records})
        pipe = StitchPipeline(deepcopy(cfg))
        for profile, options in PROFILES.items():
            pipe.config["inpainting"].update({k: v for k, v in options.items() if k != "blend_opacity"})
            pipe.config["blend"]["ai_opacity"] = float(options["blend_opacity"])
            for case in cases:
                row = {"id": case["id"], "profile": profile, "status": "failed"}
                folder = OUTPUT / profile / case["id"]
                try:
                    if case["id"] not in prepared:
                        raise ValueError("registration failed")
                    result = pipe.run(case["image_a"], case["image_b"], prepared=prepared[case["id"]], use_ai=True)
                    base = prepared[case["id"]].repair_base_image
                    outside = result.mask.soft_mask == 0
                    assert (result.final_image[outside] == base[outside]).all()
                    row.update(status="success", metrics=result.metrics,
                               quality=quality({profile: result.final_image}, case, result, ev),
                               outside_mask_unchanged=True)
                    save_run(folder, result, pipe.config, extra={"case": case, "profile": profile})
                    make_comparison(folder / "comparison.png", {"Structural": base, profile: result.final_image}, result.mask.bbox)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    save_failure(folder, exc, pipe.config)
                    traceback.print_exc()
                records.append(row)
                write_json(folder / "evaluation.json", row)
                write_json(OUTPUT / "progress-hybrid-scale-v11.json", {"records": records})
        report["status"] = "completed_with_failures" if any(r["status"] != "success" for r in records) else "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(records=records, elapsed_seconds=time.perf_counter() - started)
        write_json(OUTPUT / "report-hybrid-scale-v11.json", report)
        with zipfile.ZipFile("/kaggle/working/hybrid-scale-v11-bundle.zip", "w", zipfile.ZIP_STORED) as z:
            for p in OUTPUT.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(OUTPUT.parent))


if __name__ == "__main__":
    main()
