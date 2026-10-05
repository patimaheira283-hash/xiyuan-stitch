"""A fixed six-case pilot: geometry, masked edit, verified pixel lock."""
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
from PIL import Image

from .core import ROOT, artifact_root, file_hash, load_provider, sanitize, write_json, evaluate
from .experiment import read, read_bgr, save_bgr
from .responses_client import build_masked_edit_request, generate

BASE = ROOT / "outputs/fusion-experiments/20260909-225659"
SELECTED = [1, 4, 9, 15, 22, 23]
PROMPT = """Perform a conservative localized repair of the FIRST image, which is an already geometrically registered panorama, not an unstitched photo pair.
Its supplied alpha mask has transparent pixels ONLY around the stitching seam; opaque pixels are locked. Correct ghosted double edges, small breaks in lines, and exposure discontinuities within that narrow editable band, making transitions meet the unchanged surrounding pixels.
The second and third images are the original photographs for checking scene content. Do NOT stitch them again, change perspective, zoom, straighten the panorama, crop, expand, remove black borders, brighten the whole image, or modify the canvas. Preserve the exact first-image canvas and every pixel outside the mask as closely as possible. Black regions and outer padding are intentionally locked, not holes to fill.
Preserve text, numbers, repeated grid/brick patterns, object counts, and scene geometry. If the masked region already looks correct, leave it unchanged. Do not beautify or invent missing scene content. Return ONE edited first image, same dimensions and registration as the first input, without any markings."""


def seam_band(cell):
    """Reconstruct seam ownership using the frozen H/gains, without changing baseline."""
    r = read(cell / "traditional.json")
    base = read_bgr(cell / "traditional.png")
    height, width = base.shape[:2]
    x, y, _, _ = r["crop"]
    to_crop = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], float)
    ta = to_crop @ np.array(r["T_a_to_canvas"])
    tb = ta @ np.array(r["H_b_to_a"])
    a, b = read_bgr(cell / "left.png"), read_bgr(cell / "right.png")
    warps = [cv2.warpPerspective(im, t, (width, height)) for im, t in [(a, ta), (b, tb)]]
    supports = [cv2.warpPerspective(np.full(im.shape[:2], 255, np.uint8), t, (width, height), flags=cv2.INTER_NEAREST) for im, t in [(a, ta), (b, tb)]]
    warps[1] = np.clip(warps[1].astype(float) * np.array(r["gain_bgr"]), 0, 255).astype(np.uint8)
    scale = min(1., np.sqrt(100000 / (width * height)))
    size = (round(width * scale), round(height * scale))
    cuts = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD").find(
        [cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype(np.float32) for im in warps], [(0, 0), (0, 0)],
        [cv2.UMat(cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)) for m in supports])
    cuts = [cv2.resize(m.get() if hasattr(m, "get") else m, (width, height), interpolation=cv2.INTER_NEAREST) for m in cuts]
    valid = np.asarray(Image.open(cell / "traditional-mask.png")) > 0
    owner = np.zeros((height, width), np.uint8)
    owner[cuts[0] > 0] = 1
    owner[cuts[1] > 0] = 2
    owner[(owner == 0) & (supports[0] > 0)] = 1
    owner[(owner == 0) & (supports[1] > 0)] = 2
    a_region = (owner == 1).astype(np.uint8)
    b_region = (owner == 2).astype(np.uint8)
    boundary = (cv2.dilate(a_region, np.ones((3, 3), np.uint8)) > 0) & (cv2.dilate(b_region, np.ones((3, 3), np.uint8)) > 0) & valid
    if not boundary.any():
        raise ValueError("No seam between the two registered inputs")
    radius = max(12, round(min(width, height) * .035))
    distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    band = (distance <= radius) & valid
    return band, {"radius_native_pixels": radius, "editable_pixels": int(band.sum()), "valid_pixels": int(valid.sum()),
                  "editable_fraction": float(band.sum() / valid.sum()), "seam_boundary_pixels": int(boundary.sum())}


def prepare():
    spec = read(BASE / "experiment.json")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = BASE / ("hybrid-" + stamp)
    folder.mkdir()
    cases = []
    for c in spec["cases"]:
        if c["number"] not in SELECTED:
            continue
        source, cell = BASE / c["folder"], folder / c["folder"]
        cell.mkdir()
        for name in ["left.png", "right.png", "traditional.png", "gpt.png", "traditional-mask.png"]:
            shutil.copyfile(source / name, cell / name)
        if c["reference"]:
            shutil.copyfile(source / c["reference"], cell / "reference.png")
        band, mask_info = seam_band(source)
        Image.fromarray(band.astype(np.uint8) * 255).save(cell / "editable.png")
        base = Image.open(cell / "traditional.png").convert("RGB")
        size = (1536, 1024) if base.width >= base.height else (1024, 1536)
        scale = min(size[0] / base.width, size[1] / base.height)
        fit = (round(base.width * scale), round(base.height * scale))
        offset = ((size[0] - fit[0]) // 2, (size[1] - fit[1]) // 2)
        canvas = Image.new("RGB", size, (0, 0, 0))
        canvas.paste(base.resize(fit, Image.Resampling.LANCZOS), offset)
        canvas.save(cell / "canvas.png")
        mask_band = Image.fromarray(band.astype(np.uint8) * 255).resize(fit, Image.Resampling.NEAREST)
        editable = Image.new("L", size, 0)
        editable.paste(mask_band, offset)
        rgba = Image.new("RGBA", size, (255, 255, 255, 255))
        rgba.putalpha(Image.fromarray(255 - np.asarray(editable)))
        rgba.save(cell / "mask.png")
        overlay = np.array(canvas)
        m = np.asarray(editable) > 0
        overlay[m] = (overlay[m] * .5 + np.array([255, 60, 60]) * .5).astype(np.uint8)
        Image.fromarray(overlay).save(cell / "mask-guide.png")
        row = {"number": c["number"], "folder": c["folder"], "case_id": c["case_id"], "title": c["title"],
               "reference": c["reference"], "original_comparison": c["folder"], "canvas_size": list(size), "fit_size": list(fit),
               "offset": list(offset), "base_size": list(base.size), "mask": mask_info,
               "input_sha256": {name: file_hash(cell / name) for name in ["left.png", "right.png", "traditional.png", "canvas.png", "mask.png", "editable.png"]}}
        cases.append(row)
    write_json(folder / "experiment.json", {"id": stamp, "type": "masked_geometry_pixel_lock_v1", "source_experiment": str(BASE),
        "selection": "Six pilot cases fixed before hybrid generation: architecture, ceiling, guardrails, station, brick, text grid",
        "cases": cases, "prompt": PROMPT, "provider": load_provider(spec["provider"]["id"]).public(),
        "model": spec["model_requested"], "mainline_model": spec["mainline_model"], "quality": "high",
        "geometry": "Reuse frozen baseline H and original traditional output; no per-case re-estimation",
        "mask_recipe": "Graph-cut ownership boundary +/-3.5% of native short side (minimum12px), clipped to valid support",
        "composition": "Fit first image on standard orientation canvas; masked edit with original pair as references; SIFT protected-area homography to align returned edit; 1/3-radius inner feather; original baseline outside band copied byte-for-byte",
        "documentation": ["https://developers.openai.com/api/docs/guides/image-generation#edit-an-image-using-a-mask", "https://developers.openai.com/api/reference/resources/responses/methods/create"]})
    write_json(BASE / "hybrid-latest.json", {"folder": str(folder)})
    print(json.dumps({"folder": str(folder), "cases": [{"n": c["number"], "editable_fraction": round(c["mask"]["editable_fraction"], 3)} for c in cases]}), flush=True)
    return folder


def protected_registration(base, generated, protected):
    """Estimate return-to-canvas geometry only from protected base pixels."""
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), protected.astype(np.uint8) * 255)
    kb, db = sift.detectAndCompute(cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or min(len(da), len(db)) < 2:
        raise ValueError("Returned image has too few protected-region features")
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .7 * p[1].distance]
    if len(matches) < 24:
        raise ValueError("Fewer than 24 protected-region matches; edit not applied")
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(20260910)
    h, inside = cv2.findHomography(pb, pa, cv2.RANSAC, 3., maxIters=10000, confidence=.999)
    if h is None or inside is None:
        raise ValueError("Cannot align the returned image to protected pixels")
    inside = inside.ravel().astype(bool)
    height, width = base.shape[:2]
    span = np.ptp(pa[inside], axis=0) / [width, height] if inside.any() else np.zeros(2)
    if inside.sum() < 20 or inside.mean() < .25 or np.any(span < .3):
        raise ValueError("Protected-region registration is weak or spatially concentrated")
    gh, gw = generated.shape[:2]
    corners = cv2.perspectiveTransform(np.float32([[[0,0],[gw,0],[gw,gh],[0,gh]]]), h)[0]
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners) or not .2 < abs(cv2.contourArea(corners))/(width*height) < 5:
        raise ValueError("Implausible returned-image geometry")
    predicted = cv2.perspectiveTransform(pb[:, None], h)[:, 0]
    error = float(np.median(np.linalg.norm(predicted[inside]-pa[inside], axis=1)))
    warped = cv2.warpPerspective(generated, h, (width, height))
    valid = cv2.warpPerspective(np.full((gh, gw), 255, np.uint8), h, (width, height), flags=cv2.INTER_NEAREST) > 0
    return warped, valid, {"H_generated_to_canvas": h.tolist(), "matches": len(matches), "inliers": int(inside.sum()), "inlier_ratio": float(inside.mean()), "inlier_span_xy": span.tolist(), "median_reprojection_error": error}


def lock_composite(base, candidate, band, radius):
    """Copy protected pixels exactly; feather only inside the editable band."""
    if base.shape != candidate.shape or band.shape != base.shape[:2]:
        raise ValueError("Composite geometry does not match")
    distance = cv2.distanceTransform(band.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    weight = np.clip(distance / max(3, radius / 3), 0, 1)
    output = np.rint(base.astype(float) * (1-weight[..., None]) + candidate.astype(float) * weight[..., None]).clip(0,255).astype(np.uint8)
    output[~band] = base[~band]
    assert np.array_equal(output[~band], base[~band])
    return output, weight


def check_new_black_regions(base, candidate, band):
    """A narrow degeneration check, not a semantic quality or accuracy score."""
    unexpected = band & (base.max(axis=2) > 40) & (candidate.max(axis=2) < 8)
    fraction = float(unexpected.sum() / max(1, band.sum()))
    return {"passed": fraction <= .05, "new_black_pixels": int(unexpected.sum()), "new_black_fraction_of_band": fraction,
            "threshold": .05, "definition": "base max RGB >40 becomes candidate max RGB <8; reject if >5% of editable band"}


def accept_or_fallback(folder, case):
    cell = folder / case["folder"]
    meta_path = cell / "hybrid.json"
    if not meta_path.exists():
        generation = read(cell / "generation.json") if (cell / "generation.json").exists() else {}
        if generation.get("status") != "failed":
            return {"status": "pending"}
        meta = {"status": "generation_failed"}
    else:
        meta = read(meta_path)
    r = {"status": "fallback", "accepted": False, "rule_version": "black-region-check-v1",
         "note": "Added after observing the brick failure; applied uniformly to all six retained candidates. This catches degeneration, not all hallucination or seam defects."}
    if meta["status"] == "applied":
        base, candidate = read_bgr(cell / "traditional.png"), read_bgr(cell / "hybrid.png")
        band = np.asarray(Image.open(cell / "editable.png")) > 0
        r["black_region_check"] = check_new_black_regions(base, candidate, band)
        if meta["protected_pixel_changes"] == 0 and r["black_region_check"]["passed"]:
            r.update(status="accepted", accepted=True)
        else:
            r["reason"] = "接缝内出现大块新增黑色区域，拒绝采用，保留传统结果。"
    else:
        r["reason"] = "生图请求失败，保留传统结果。" if meta["status"] == "generation_failed" else "模型候选未通过几何对齐验收，保留传统结果。"
    source = cell / ("hybrid.png" if r["accepted"] else "traditional.png")
    shutil.copyfile(source, cell / "safe.png")
    r["output_sha256"] = file_hash(cell / "safe.png")
    if case["reference"]:
        r["evaluation"] = evaluate(cell / "safe.png", cell / "reference.png")
    write_json(cell / "acceptance.json", r)
    return r


def compose(folder, case):
    cell = folder / case["folder"]
    start = time.perf_counter()
    r = {"status": "not_applied", "source_case": case["number"]}
    base = read_bgr(cell / "traditional.png")
    band = np.asarray(Image.open(cell / "editable.png")) > 0
    try:
        canvas = read_bgr(cell / "canvas.png")
        alpha = np.asarray(Image.open(cell / "mask.png"))[:, :, 3]
        valid_canvas = (np.max(canvas, axis=2) > 0)
        protected = (alpha == 255) & valid_canvas
        raw_generated = read_bgr(cell / "raw.png")
        if raw_generated.shape == canvas.shape:
            native_delta = np.abs(raw_generated.astype(np.int16)-canvas.astype(np.int16))
            r["raw_native_protected_mae"] = float(native_delta[protected].mean())
            r["raw_native_protected_changed_fraction"] = float(np.any(native_delta[protected] != 0, axis=1).mean())
        candidate, coverage, registration = protected_registration(canvas, raw_generated, protected)
        x, y = case["offset"]; w, h = case["fit_size"]
        candidate = cv2.resize(candidate[y:y+h, x:x+w], tuple(case["base_size"]), interpolation=cv2.INTER_LANCZOS4)
        coverage = cv2.resize(coverage[y:y+h, x:x+w].astype(np.uint8), tuple(case["base_size"]), interpolation=cv2.INTER_NEAREST) > 0
        editable_coverage = float(coverage[band].mean())
        if editable_coverage < .995:
            raise ValueError("Returned image does not cover at least 99.5% of the editable band")
        candidate[~coverage] = base[~coverage]
        output, weights = lock_composite(base, candidate, band, case["mask"]["radius_native_pixels"])
        save_bgr(cell / "aligned-candidate.png", candidate)
        Image.fromarray(np.rint(weights * 255).astype(np.uint8)).save(cell / "blend-weights.png")
        save_bgr(cell / "hybrid.png", output)
        delta = np.abs(output.astype(np.int16) - base.astype(np.int16))
        raw_delta = np.abs(candidate.astype(np.int16) - base.astype(np.int16))
        valid_photo = np.asarray(Image.open(cell / "traditional-mask.png")) > 0
        r.update(status="applied", registration=registration, editable_coverage=editable_coverage,
                 protected_pixel_changes=int(np.any(delta[~band] != 0, axis=1).sum()), protected_max_channel_delta=int(delta[~band].max()),
                 editable_mae=float(delta[band].mean()), raw_aligned_protected_mae=float(raw_delta[~band].mean()),
                 raw_aligned_protected_photo_mae=float(raw_delta[(~band) & valid_photo].mean()),
                 changed_pixel_fraction=float(np.any(delta != 0,axis=2).mean()), size_actual=list(case["base_size"]), output_sha256=file_hash(cell / "hybrid.png"))
        if case["reference"]:
            r["evaluation"] = evaluate(cell / "hybrid.png", cell / "reference.png")
    except Exception as exc:
        # A rejection is visible; do not imply the original baseline is an AI improvement.
        r["error"] = sanitize(str(exc))
    r["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    write_json(cell / "hybrid.json", r)
    return r


def run_one(folder, spec, c):
    cell = folder / c["folder"]
    path = cell / "generation.json"
    if path.exists():
        return read(path)
    provider = load_provider(spec["provider"]["id"])
    run_id = f"hybrid-{spec['id']}-{c['folder']}"
    artifact = artifact_root() / ("xiyuan-" + run_id + "-raw.png")
    artifact.parent.mkdir(exist_ok=True, parents=True)
    prompt = spec["prompt"] + f"\nThe exact output canvas is {c['canvas_size'][0]} x {c['canvas_size'][1]} pixels."
    (cell / "prompt.txt").write_text(prompt, encoding="utf-8")
    payload = build_masked_edit_request([cell / "canvas.png", cell / "left.png", cell / "right.png"], cell / "mask.png", prompt,
                 spec["mainline_model"], spec["model"], spec["quality"], "x".join(map(str, c["canvas_size"])))
    preview = json.loads(json.dumps(payload))
    for i in preview["input"][0]["content"]:
        if i["type"] == "input_image": i["image_url"] = "[local PNG bytes omitted]"
    preview["tools"][0]["input_image_mask"]["image_url"] = "[local RGBA mask bytes omitted]"
    write_json(cell / "request-preview.json", preview)
    r = {"status": "running", "id": run_id, "provider": provider.public(), "model_requested": spec["model"],
         "mainline_model_requested": spec["mainline_model"], "quality_requested": spec["quality"], "size_requested": c["canvas_size"],
         "native_mask_sent": True, "input_count": 3, "created_at": datetime.now().astimezone().isoformat(), "backend_identity_verified": False, "retry_count": 0}
    write_json(path, r)
    start = time.perf_counter()
    try:
        r.update(generate(provider, payload, artifact))
        with Image.open(artifact) as im: r["size_actual"] = list(im.size)
        shutil.copyfile(artifact, cell / "raw.png")
        r.update(status="succeeded", artifact_path=artifact.as_posix(), output_sha256=file_hash(artifact))
    except Exception as exc:
        r.update(status="failed", error=sanitize(str(exc), provider))
    r["elapsed_seconds"] = round(time.perf_counter()-start, 3)
    write_json(path, r)
    return r


def batch(folder, numbers=None):
    cv2.setNumThreads(2)
    spec = read(folder / "experiment.json")
    selected = [c for c in spec["cases"] if numbers is None or c["number"] in numbers]
    # Two workers, no implicit paid retry; a schema failure stops the next wave.
    for i in range(0, len(selected), 2):
        stop = False
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(run_one, folder, spec, c): c for c in selected[i:i+2]}
            for future in as_completed(jobs):
                c = jobs[future]; r = future.result()
                result = compose(folder, c) if r["status"] == "succeeded" else None
                acceptance = accept_or_fallback(folder, c)
                print(json.dumps({"n": c["number"], "generation": r["status"], "seconds": r.get("elapsed_seconds"),
                                  "composition": result and result["status"], "acceptance": acceptance["status"], "error": r.get("error") or (result and result.get("error"))}, ensure_ascii=False), flush=True)
                if r["status"] != "succeeded": stop = True
        if stop: break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "compose", "accept"])
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--numbers", nargs="+", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        cv2.setNumThreads(2)
        prepare()
    else:
        folder = args.folder or Path(read(BASE / "hybrid-latest.json")["folder"])
        if args.action == "run": batch(folder, args.numbers)
        else:
            for c in read(folder / "experiment.json")["cases"]:
                if args.numbers is None or c["number"] in args.numbers:
                    result = accept_or_fallback(folder, c) if args.action == "accept" else compose(folder, c)
                    print(json.dumps({"n": c["number"], **result}, ensure_ascii=False))
