"""Tune protected RAFT confidence gates while keeping clear fusion fixed."""
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
from xiyuan_mvp.neural_alignment import repair_alignment
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import environment, save_failure, save_run, write_json

PROFILES = {
    "consistency_100_flow_16": {"consistency": 1.0, "max_flow": 16},
    "consistency_100_flow_24": {"consistency": 1.0, "max_flow": 24},
    "consistency_150_flow_16": {"consistency": 1.5, "max_flow": 16},
    "consistency_150_flow_24": {"consistency": 1.5, "max_flow": 24},
    "consistency_200_flow_24": {"consistency": 2.0, "max_flow": 24},
    "consistency_200_flow_32": {"consistency": 2.0, "max_flow": 32},
}

def main():
    started = time.perf_counter(); records = []
    report = {"status": "running", "experiment": "protected RAFT confidence gate tuning",
              "protocol": {"references_used_by_repair": False, "clear_fusion_fixed": True,
                           "profiles": PROFILES}}
    try:
        assert torch.cuda.is_available()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1, 8 * 1024**3 / total))
        report.update(gpu=torch.cuda.get_device_name(0), actual_vram_bytes=total,
                      allocator_limit_bytes=min(total, 8 * 1024**3), environment=environment())
        cfg = load_config()
        cfg["registration"].update(method="loftr", loftr_device="cuda", loftr_max_size=640)
        cfg["clear_fusion"].update(enabled=True, mode="adaptive_detail", guard=True)
        cfg["neural_alignment"].update(device="cuda", method="raft_large", preserve_detail=True,
                                       max_size=512, iterations=12, exposure=True)
        cases = inputs() + create_cases(DATA / "deformation")
        write_json(OUTPUT / "inputs-raft-gate-v15.json", {"cases": cases, "profiles": PROFILES, "config": cfg})
        evaluator = LPIPSEvaluator(); prepared = {}
        for case in cases:
            try:
                prepared[case["id"]] = StitchPipeline(cfg).run(case["image_a"], case["image_b"])
            except Exception as exc:
                row = {"id": case["id"], "profile": "clear_structural_only", "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"}
            else:
                base = prepared[case["id"]]
                row = {"id": case["id"], "profile": "clear_structural_only", "status": "success",
                       "metrics": base.metrics,
                       "quality": quality({"Structural": base.repair_base_image}, case, base, evaluator)}
            records.append(row); write_json(OUTPUT / "progress-raft-gate-v15.json", {"records": records})
        for profile, options in PROFILES.items():
            neural = {**cfg["neural_alignment"], **options}
            for case in cases:
                folder = OUTPUT / profile / case["id"]
                row = {"id": case["id"], "profile": profile, "status": "failed"}
                try:
                    if case["id"] not in prepared: raise ValueError("registration failed")
                    base = prepared[case["id"]]
                    image, metrics = repair_alignment(base.registration, base.repair_base_image,
                                                       base.mask, neural, method="raft_large")
                    outside = base.mask.soft_mask == 0
                    assert (image[outside] == base.repair_base_image[outside]).all()
                    row.update(status="success", metrics=metrics,
                               quality=quality({profile: image}, case, base, evaluator),
                               outside_mask_unchanged=True)
                    save_run(folder, base, cfg, extra={"case": case, "profile": profile, "repair_metrics": metrics})
                    make_comparison(folder / "comparison.png", {"Structural": base.repair_base_image, profile: image}, base.mask.bbox)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"; save_failure(folder, exc, cfg); traceback.print_exc()
                records.append(row); write_json(folder / "evaluation.json", row)
                write_json(OUTPUT / "progress-raft-gate-v15.json", {"records": records})
        report["status"] = "completed_with_failures" if any(r["status"] != "success" for r in records) else "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        report.update(records=records, elapsed_seconds=time.perf_counter() - started)
        write_json(OUTPUT / "report-raft-gate-v15.json", report)
        with zipfile.ZipFile("/kaggle/working/raft-gate-v15-bundle.zip", "w", zipfile.ZIP_STORED) as z:
            for p in OUTPUT.rglob("*"):
                if p.is_file(): z.write(p, p.relative_to(OUTPUT.parent))

if __name__ == "__main__": main()
