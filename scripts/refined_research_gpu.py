from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback
import zipfile

import cv2
import numpy as np
import torch

from scripts.research_gpu import inputs, OUTPUT, ROOT, DATA
from scripts.deformation_cases import create_cases
from xiyuan_mvp.benchmark import LPIPSEvaluator, masked_metrics, make_comparison
from xiyuan_mvp.config import load_config
from xiyuan_mvp.image_io import read_image, write_image
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import environment, save_run, save_failure, write_json

PROFILES={
    "raw_canny":{"controlnet":"canny","controlnet_mask_mode":"none","use_lcm":False,"steps":20,"strength":0.25,"guidance_scale":3.5,"raw_base":True},
    "refined_canny":{"controlnet":"canny","controlnet_mask_mode":"none","use_lcm":False,"steps":20,"strength":0.25,"guidance_scale":3.5},
    # Candidate controls: only the Canny conditioning changes; all diffusion
    # and structural settings stay fixed for an apples-to-apples GPU run.
    "refined_context_canny":{"controlnet":"canny","controlnet_mask_mode":"context_edges","use_lcm":False,"steps":20,"strength":0.25,"guidance_scale":3.5},
    "refined_faded_canny":{"controlnet":"canny","controlnet_mask_mode":"faded_edges","controlnet_edge_fade":8,"controlnet_edge_floor":0.15,"use_lcm":False,"steps":20,"strength":0.25,"guidance_scale":3.5},
    "refined_canny_010":{"controlnet":"canny","controlnet_mask_mode":"none","use_lcm":False,"steps":20,"strength":0.25,"guidance_scale":3.5,"blend_opacity":0.10},
    "refined_canny_005":{"controlnet":"canny","controlnet_mask_mode":"none","use_lcm":False,"steps":20,"strength":0.25,"guidance_scale":3.5,"blend_opacity":0.05},
    "refined_faded_canny_010":{"controlnet":"canny","controlnet_mask_mode":"faded_edges","controlnet_edge_fade":8,"controlnet_edge_floor":0.15,"use_lcm":False,"steps":20,"strength":0.25,"guidance_scale":3.5,"blend_opacity":0.10},
    "refined_tile":{"controlnet":"tile","controlnet_mask_mode":"none","use_lcm":False,"steps":20,"strength":0.25,"guidance_scale":3.5},
    "refined_tile_lcm":{"controlnet":"tile","controlnet_mask_mode":"none","use_lcm":True,"steps":8,"strength":0.5,"guidance_scale":1.0},
}


def quality(images,case,result,evaluator):
    if not case.get("reference"):
        return None
    reference=read_image(case["reference"])
    h,w=result.final_image.shape[:2]
    transform=result.registration.canvas_transform@np.asarray(case["reference_to_a"])
    ref=cv2.warpPerspective(reference,transform,(w,h))
    valid=cv2.warpPerspective(np.full(reference.shape[:2],255,np.uint8),transform,(w,h))
    valid=cv2.erode(valid,np.ones((7,7),np.uint8))
    seam=cv2.bitwise_and(valid,result.mask.binary_mask)
    scores={}
    for name,image in images.items():
        scores[name]={}
        for region,mask in (("full",valid),("seam",seam)):
            scores[name][region]={**masked_metrics(image,ref,mask),"lpips":evaluator.evaluate(image,ref,mask)}
    return scores


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    report={"status":"running","experiment":"v2 development: photometric/local geometric preparation and restrained diffusion"}
    records=[]
    try:
        assert torch.cuda.is_available()
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1,8*1024**3/total))
        report.update(gpu=torch.cuda.get_device_name(0),actual_vram_bytes=total,
                      allocator_limit_bytes=min(total,8*1024**3),environment=environment(),
                      budget_note="8 GiB allocator cap on T4; not a native 8GB card.")
        config=load_config()
        config["registration"].update(method="loftr",loftr_device="cuda",loftr_max_size=640)
        config["refinement"]["enabled"]=True
        config["blend"]["ai_opacity"]=0.2
        config["inpainting"].update(device="cuda",local_files_only=False,variant="fp16",
                                   controlnet_scale=0.8,cache_dir="/kaggle/temp/xiyuan-hf-cache")
        cases=inputs()+create_cases(DATA/"deformation")
        write_json(OUTPUT/"inputs-v2.json",{"cases":cases})
        evaluator=LPIPSEvaluator()
        prepared={}
        for case in cases:
            try:
                baseline=StitchPipeline(config).run(case["image_a"],case["image_b"])
                prepared[case["id"]]=baseline
                scores=quality({"Feather":baseline.traditional_image,"Poisson":baseline.poisson_image,
                                "Structural":baseline.repair_base_image},case,baseline,evaluator)
                record={"id":case["id"],"profile":"structural_only","status":"success","quality":scores,"metrics":baseline.metrics}
                save_run(OUTPUT/"structural_only"/case["id"],baseline,config,extra={"case":case})
            except Exception as exc:
                record={"id":case["id"],"profile":"structural_only","status":"failed","error":f"{type(exc).__name__}: {exc}"}
                traceback.print_exc()
            records.append(record)
            write_json(OUTPUT/"progress-v2.json",{"records":records})
            print("BASE",json.dumps(record),flush=True)
        pipe=StitchPipeline(config)
        for profile,options in PROFILES.items():
            ai_options={k:v for k,v in options.items() if k not in ("raw_base","blend_opacity")}
            pipe.config["inpainting"].update(ai_options)
            pipe.config["blend"]["ai_opacity"] = float(options.get("blend_opacity", 0.2))
            for case in cases:
                if case["id"] not in prepared:
                    continue
                baseline=prepared[case["id"]]
                if options.get("raw_base"):
                    baseline=replace(baseline,repair_base_image=baseline.traditional_image,
                                     metrics={**baseline.metrics,"refinement_enabled":False})
                folder=OUTPUT/profile/case["id"]
                record={"id":case["id"],"profile":profile,"status":"failed"}
                try:
                    result=pipe.run(case["image_a"],case["image_b"],prepared=baseline,use_ai=True)
                    outside=result.mask.soft_mask==0
                    assert np.array_equal(result.final_image[outside],baseline.repair_base_image[outside])
                    scores=quality({profile:result.final_image},case,result,evaluator)
                    record.update(status="success",quality=scores,metrics=result.metrics,
                                  outside_mask_unchanged=True,ai_opacity=pipe.config["blend"]["ai_opacity"])
                    save_run(folder,result,pipe.config,extra={"case":case,"profile":profile})
                    images={"Feather":baseline.traditional_image,"Poisson":baseline.poisson_image,
                            "Structural":prepared[case["id"]].repair_base_image,profile:result.final_image}
                    make_comparison(folder/"comparison.png",images,result.mask.bbox)
                except Exception as exc:
                    record["error"]=f"{type(exc).__name__}: {exc}"
                    save_failure(folder,exc,pipe.config)
                    traceback.print_exc()
                records.append(record)
                write_json(folder/"evaluation.json",record)
                write_json(OUTPUT/"progress-v2.json",{"records":records})
                print("RESULT",json.dumps(record),flush=True)
        report["status"]="completed_with_failures" if any(r["status"]!="success" for r in records) else "completed"
    except Exception as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        report.update(records=records,elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT/"report-v2.json",report)
        bundle=Path("/kaggle/working/research-v2-bundle.zip")
        with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as archive:
            for p in OUTPUT.rglob("*"):
                if p.is_file():
                    archive.write(p,p.relative_to(OUTPUT.parent))
        print("BUNDLE",bundle.stat().st_size,flush=True)


if __name__=="__main__":
    main()
