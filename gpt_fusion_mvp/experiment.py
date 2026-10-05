"""A frozen paired experiment: classical stitching versus direct image generation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from .core import ROOT, LIBRARY, RUNS, MODEL, MAINLINE_MODEL, PROMPT, file_hash, list_runs, load_provider, run_fusion, write_json, evaluate
from .materials import get_manifest

EXPERIMENTS = ROOT / "outputs" / "fusion-experiments"
EXCLUDE = {"spw-10-efab8e6e", "spw-11-631df5c6", "spw-12-f66932e8", "lpc-school-0c1c5210", "rew-rew-wall-de776f7d"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    library = get_manifest()
    indexed = {c["id"]: c for c in library["cases"]}
    smoke = read(LIBRARY / "smoke-suite.json")["case_ids"]
    selected = [indexed[c] for c in smoke if c not in EXCLUDE]
    assert len(selected) == 20
    selected += [indexed[c] for c in ["coffee-clean", "brick-exposure", "text-grid-clean"]]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = EXPERIMENTS / stamp
    folder.mkdir(parents=True)
    records = []
    for i, case in enumerate(selected, 1):
        cell = folder / f"{i:02d}"
        cell.mkdir()
        inputs = []
        for key in ("left", "right"):
            source = LIBRARY / case[key]
            with Image.open(source) as photo:
                photo.load()
                image = ImageOps.exif_transpose(photo).convert("RGB")
                original_size = list(image.size)
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                path = cell / f"{key}.png"
                image.save(path)
            inputs.append({"source": str(source), "source_sha256": file_hash(source), "file": path.name,
                           "sha256": file_hash(path), "original_size": original_size, "size": list(image.size)})
        if case.get("reference"):
            shutil.copyfile(LIBRARY / case["reference"], cell / "reference.png")
        records.append({"number": i, "case_id": case["id"], "title": case["title"], "source_id": case["source_id"],
                        "kind": case["kind"], "scene": case.get("scene", case["source_id"]), "inputs": inputs,
                        "prompt": case.get("prompt", PROMPT), "reference": "reference.png" if case.get("reference") else None,
                        "tags": case.get("tags", ["受控参考"]), "folder": f"{i:02d}", "run_id": f"bench-{stamp}-{i:02d}"})
    spec = {"id": stamp, "created_at": datetime.now().astimezone().isoformat(), "provider": load_provider().public(),
            "model_requested": MODEL, "mainline_model": MAINLINE_MODEL, "quality": "high", "size": "1536x1024",
            "input_preparation": "EXIF orientation + RGB PNG; longest side <=1600 without upscaling; identical prepared pixels for both methods",
            "selection": "20 diverse real scene folders from the 25-case recommended suite (remove 3 redundant SPW outdoor views, LPC school, REW wall); 3 controlled cases appended before outcomes",
            "classical": "SIFT 6000 + symmetric ratio .75 + RANSAC homography (3px, confidence .999, 10000 iters), robust overlap color gain, graph-cut seam at <=0.1 MP, 5-band blending at input resolution; one fixed recipe for all cases",
            "cases": records}
    write_json(folder / "experiment.json", spec)
    write_json(EXPERIMENTS / "latest.json", {"folder": str(folder)})
    print(str(folder), flush=True)
    return folder


def read_bgr(path):
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def save_bgr(path, image):
    Image.fromarray(image[:, :, ::-1]).save(path)


def sift_matches(a, b):
    sift = cv2.SIFT_create(nfeatures=6000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or min(len(da), len(db)) < 2:
        raise ValueError("Not enough SIFT features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(da, db, k=2)
    reverse = matcher.knnMatch(db, da, k=2)
    backward = {(v[0].trainIdx, v[0].queryIdx) for v in reverse if len(v)==2 and v[0].distance < .75*v[1].distance}
    good = [v[0] for v in forward if len(v)==2 and v[0].distance < .75*v[1].distance and (v[0].queryIdx,v[0].trainIdx) in backward]
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    return pa, pb, len(ka), len(kb)


def classical(folder, case):
    cell = folder / case["folder"]
    started = time.perf_counter()
    record = {"status": "failed", "method": "SIFT-RANSAC-GraphCut-Multiband", "opencv": cv2.__version__}
    try:
        a, b = read_bgr(cell / "left.png"), read_bgr(cell / "right.png")
        pa, pb, na, nb = sift_matches(a, b)
        record.update(keypoints=[na, nb], matches=len(pa))
        if len(pa) < 8:
            raise ValueError("Too few mutual feature matches")
        cv2.setRNGSeed(20260909)
        h, mask = cv2.findHomography(pb, pa, cv2.RANSAC, 3, maxIters=10000, confidence=.999)
        if h is None or mask is None or int(mask.sum()) < 8:
            raise ValueError("Homography rejected: fewer than 8 inliers")
        inside = mask.ravel().astype(bool)
        projected = cv2.perspectiveTransform(pb[:, None], h)[:, 0]
        record.update(inliers=int(inside.sum()), inlier_ratio=float(inside.mean()),
                      median_reprojection_error=float(np.median(np.linalg.norm(projected[inside]-pa[inside], axis=1))), H_b_to_a=h.tolist())
        ha, wa = a.shape[:2]; hb, wb = b.shape[:2]
        corners_a = np.float32([[0,0],[wa,0],[wa,ha],[0,ha]])
        corners_b = cv2.perspectiveTransform(np.float32([[[0,0],[wb,0],[wb,hb],[0,hb]]]), h)[0]
        if not np.isfinite(corners_b).all() or not cv2.isContourConvex(corners_b):
            raise ValueError("Invalid projected quadrilateral")
        area = abs(cv2.contourArea(corners_b)) / (wb*hb)
        if not .15 < area < 6:
            raise ValueError("Implausible homography scale")
        corners = np.vstack([corners_a, corners_b])
        low, high = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
        w, height = (high-low).astype(int)
        if min(w, height) < 10 or w*height > 16_000_000 or max(w,height) > 7000:
            raise ValueError("Panorama canvas outside fixed limits")
        transform = np.array([[1,0,-low[0]],[0,1,-low[1]],[0,0,1]], float)
        warp_a = cv2.warpPerspective(a, transform, (w,height))
        warp_b = cv2.warpPerspective(b, transform @ h, (w,height))
        mask_a = cv2.warpPerspective(np.full((ha,wa),255,np.uint8), transform, (w,height), flags=cv2.INTER_NEAREST)
        mask_b = cv2.warpPerspective(np.full((hb,wb),255,np.uint8), transform @ h, (w,height), flags=cv2.INTER_NEAREST)
        overlap = (mask_a > 0) & (mask_b > 0)
        if int(overlap.sum()) < 500:
            raise ValueError("Insufficient projected overlap")
        gains = []
        for channel in range(3):
            usable = overlap & (warp_a[:,:,channel]>20) & (warp_b[:,:,channel]>20) & (warp_a[:,:,channel]<235) & (warp_b[:,:,channel]<235)
            gain = float(np.median(warp_a[:,:,channel][usable].astype(float) / warp_b[:,:,channel][usable])) if usable.sum()>100 else 1.
            gains.append(float(np.clip(gain,.7,1.4)))
        warp_b = np.clip(warp_b.astype(float)*np.array(gains),0,255).astype(np.uint8)
        seam_scale = min(1.,np.sqrt(100000/(w*height)))
        seam_size = (max(1,round(w*seam_scale)),max(1,round(height*seam_scale)))
        seam_images = [cv2.resize(im,seam_size,interpolation=cv2.INTER_AREA).astype(np.float32) for im in [warp_a,warp_b]]
        seam_inputs = [cv2.UMat(cv2.resize(mask,seam_size,interpolation=cv2.INTER_NEAREST)) for mask in [mask_a,mask_b]]
        seam = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        found = seam.find(seam_images,[(0,0),(0,0)],seam_inputs)
        seam_masks = [cv2.resize(v.get() if hasattr(v,'get') else v,(w,height),interpolation=cv2.INTER_NEAREST) for v in found]
        record["seam_estimation_size"] = list(seam_size)
        blend = cv2.detail_MultiBandBlender(0,5)
        blend.prepare((0,0,int(w),int(height)))
        for im, cut, support in zip([warp_a,warp_b],seam_masks,[mask_a,mask_b]):
            cut = cv2.bitwise_and(cv2.dilate(cut,np.ones((3,3),np.uint8)),support)
            blend.feed(im.astype(np.int16), cut, (0,0))
        output, valid = blend.blend(None,None)
        output = np.clip(output,0,255).astype(np.uint8)
        x,y,cw,ch = cv2.boundingRect(valid)
        output = output[y:y+ch,x:x+cw]
        save_bgr(cell / "traditional.png", output)
        Image.fromarray(valid[y:y+ch,x:x+cw]).save(cell / "traditional-mask.png")
        record.update(status="succeeded", size_actual=[cw,ch], overlap_pixels=int(overlap.sum()),
                      gain_bgr=gains, output_sha256=file_hash(cell/"traditional.png"), T_a_to_canvas=transform.tolist(), crop=[x,y,cw,ch])
        if case["reference"]:
            record["evaluation"] = evaluate(cell/"traditional.png",cell/case["reference"])
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.perf_counter()-started,3)
    write_json(cell/"traditional.json",record)
    return record


def baselines(folder):
    cv2.setNumThreads(2)
    spec = read(folder/"experiment.json")
    for case in spec["cases"]:
        cell=folder/case["folder"]
        if (cell/"traditional.json").is_file():
            record=read(cell/"traditional.json")
        else:
            record=classical(folder,case)
        print(json.dumps({"number":case["number"],"status":record["status"],"seconds":record["elapsed_seconds"],"error":record.get("error")}),flush=True)


def gpt_batch(folder, workers=2, retry_failed=False):
    spec=read(folder/"experiment.json")
    earlier=list_runs()
    def one(case):
        cell=folder/case["folder"]
        path=cell/"gpt.json"
        if path.is_file():
            prior=read(path)
            if not retry_failed or prior["status"] != "failed":
                return prior
            # Exactly one explicit recovery attempt; preserve the initial failure.
            if not (cell/"gpt-initial-attempt.json").exists():
                write_json(cell/"gpt-initial-attempt.json",prior)
        run_id=case["run_id"] + ("-recovery1" if retry_failed and path.exists() else "")
        existing=RUNS/run_id/"run.json"
        if existing.is_file():
            record=read(existing)
            record["reused_existing_attempt"]=True
        else:
            matching=[r for r in earlier if r.get("status")=="succeeded" and r.get("case_id")==case["case_id"]
                      and r.get("input_sha256")==[i["sha256"] for i in case["inputs"]]
                      and r.get("model_requested")==spec["model_requested"] and r.get("quality")==spec["quality"]
                      and r.get("size_requested")==spec["size"] and r.get("mainline_model_requested")==spec["mainline_model"]
                      and r.get("provider",{}).get("id")==spec["provider"]["id"]
                      and (RUNS/r["id"]/"prompt.txt").read_text(encoding="utf-8")==case["prompt"]]
            if matching:
                # First matching completed attempt, not the best-looking output.
                record=min(matching,key=lambda r:r.get("created_at",r["id"]))
                record={**record,"reused_prior_run":True}
            else:
                record=run_fusion(cell/"left.png",cell/"right.png",case_id=case["case_id"],prompt=case["prompt"],
                    model=spec["model_requested"],mainline_model=spec["mainline_model"],quality=spec["quality"],size=spec["size"],
                    truth=cell/case["reference"] if case["reference"] else None,provider_id=spec["provider"]["id"],run_id=run_id)
        if record["status"]=="succeeded":
            shutil.copyfile(record["artifact_path"],cell/"gpt.png")
        write_json(path,record)
        return record
    completed=[]
    cases=iter(spec["cases"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={}
        for _ in range(workers):
            c=next(cases,None)
            if c:pending[pool.submit(one,c)]=c
        stopped=False
        consecutive_failures=0
        while pending:
            future=next(as_completed(pending))
            case=pending.pop(future)
            try:
                result=future.result()
            except Exception as exc:
                from .core import sanitize
                result={"status":"failed","error":sanitize(str(exc)),"elapsed_seconds":None}
                write_json(folder/case["folder"]/"gpt.json",result)
            summary={"number":case["number"],"case_id":case["case_id"],"status":result["status"],
                     "seconds":result.get("elapsed_seconds"),"reused":result.get("reused_prior_run",False),"error":result.get("error")}
            completed.append(summary)
            write_json(folder/"progress.json",{"completed":completed,"total":len(spec["cases"]),"stopped":stopped})
            print(json.dumps(summary,ensure_ascii=False),flush=True)
            error=result.get("error","").lower()
            consecutive_failures=consecutive_failures+1 if result["status"]=="failed" else 0
            if consecutive_failures >= 3:
                stopped=True
            if result["status"]=="failed" and any(token in error for token in ["http 401","http 403","http 404","quota","insufficient_quota"]):
                stopped=True
            c=next(cases,None) if not stopped else None
            if c:pending[pool.submit(one,c)]=c
        write_json(folder/"progress.json",{"completed":completed,"total":len(spec["cases"]),"stopped":stopped})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","baseline","gpt"])
    parser.add_argument("--folder",type=Path)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--retry-failed",action="store_true",help="One explicit recovery attempt; retains initial failure and never regenerates successes")
    args=parser.parse_args()
    if args.action=="prepare":prepare()
    else:
        folder=args.folder or Path(read(EXPERIMENTS/"latest.json")["folder"])
        if args.action=="baseline":baselines(folder)
        else:gpt_batch(folder,args.workers,args.retry_failed)
