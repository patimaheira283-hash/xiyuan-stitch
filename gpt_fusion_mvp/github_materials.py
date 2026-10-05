"""Download pinned public stitching inputs and build a traceable local test catalog.

No repository code is executed. Original files are retained byte-for-byte.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .core import LIBRARY, file_hash, write_json

SPECS = [
    ("spw", "tlliao/Single-perspective-warps", "TwoImage/Images/", 4, "SPW · 单视角结构"),
    ("lpc", "dut-media-lab/Image-Stitching", "Imgs/", 3, "LPC · 大视差与线条"),
    ("rew", "gain2217/Robust_Elastic_Warping", "two_views/images/", 4, "REW · 视差容忍"),
    ("ges", "flowerDuo/GES-GSP-Stitching", "Dataset/", 3, "GES-50 · 多视图结构"),
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".tif", ".tiff"}
PROVENANCE = LIBRARY / "provenance"
ORIGINALS = LIBRARY / "originals"
REAL_PROMPT = """These are two overlapping photographs of the SAME real scene.
Create one coherent stitched photograph showing the union of their fields of view.
Infer the relative position and orientation from shared visual content: input order does not imply left/right or top/bottom.
Preserve the actual scene, object count, people, text, straight architectural lines, and fine details as faithfully as possible.
Resolve overlap, parallax and exposure differences while avoiding duplicated objects, visible seams, ghosting and invented content.
Do not redesign the scene, add objects, replace textures or turn this into a collage.
Use only the two provided photographs. Return a single stitched image without captions or borders."""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def blob_hash(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def download_file(url, dest, expected_blob=None):
    if dest.is_file():
        raw = dest.read_bytes()
        if expected_blob is None or blob_hash(raw) == expected_blob:
            return
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 90), headers={"User-Agent": "Xiyuan-local-dataset-library"})
            response.raise_for_status()
            raw = response.content
            if expected_blob and blob_hash(raw) != expected_blob:
                raise ValueError(f"Git blob checksum mismatch: {dest.name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name + ".part")
            temp.write_bytes(raw)
            temp.replace(dest)
            return
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def download():
    tasks = []
    for sid, repo, prefix, depth, _ in SPECS:
        data = read_json(PROVENANCE / (repo.replace("/", "__") + ".json"))
        commit = data["tree"]["sha"]
        for entry in data["tree"]["tree"]:
            p = PurePosixPath(entry["path"])
            if str(p).startswith(prefix) and len(p.parts) == depth and (p.suffix.lower() in EXTENSIONS or str(p).endswith("STITCH-GRAPH.txt")):
                url = f"https://raw.githubusercontent.com/{repo}/{commit}/{quote(str(p))}"
                tasks.append((url, ORIGINALS / sid / str(p), entry["sha"]))
    print(f"Downloading/verifying {len(tasks)} pinned image and graph files", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_file, *task): task[1] for task in tasks}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append({"path": str(futures[future]), "error": str(exc)})
            if n % 50 == 0 or n == len(tasks):
                print(f"Files {n}/{len(tasks)}, failures {len(failures)}", flush=True)
    release = read_json(PROVENANCE / "OpenPano-release.json")
    asset = next(a for a in release["assets"] if a["name"] == "example-data.tgz")
    archive = ORIGINALS / "openpano" / asset["name"]
    download_file(asset["browser_download_url"], archive)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            p = PurePosixPath(member.name)
            if not member.isfile() or p.suffix.lower() not in EXTENSIONS | {".txt", ".md", ".cfg"}:
                continue
            dest = (ORIGINALS / "openpano" / str(p)).resolve()
            if not dest.is_relative_to((ORIGINALS / "openpano").resolve()) or p.is_absolute() or member.size > 50_000_000:
                raise ValueError("Unsafe archive member")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(member).read())
    write_json(PROVENANCE / "download-status.json", {"files_expected": len(tasks), "failures": failures,
        "openpano_archive_sha256": file_hash(archive), "openpano_download": asset["browser_download_url"]})
    print(f"Download complete. Failures: {len(failures)}", flush=True)
    if failures:
        raise RuntimeError("Some downloads failed; see provenance/download-status.json and rerun download.")


def graph_pairs(text, count, rejected=None):
    declared = re.search(r"\{images_count\s*\|\s*(\d+)", text)
    if declared is None or int(declared[1]) != count:
        raise ValueError(f"Graph/image count mismatch: expected {count}")
    pairs = set()
    for index, neighbors in re.findall(r"\{matching_graph_image_edges-(\d+)\s*\|\s*([^|]*)\|", text):
        for neighbor in re.findall(r"\d+", neighbors):
            i, j = int(index), int(neighbor)
            if not (0 <= i < count and 0 <= j < count) or i == j:
                if rejected is None:
                    raise ValueError(f"Invalid graph edge {i},{j}")
                rejected.append({"edge": [i, j], "image_count": count, "reason": "upstream_graph_index_out_of_range_or_self_edge"})
                continue
            pairs.add(tuple(sorted((i, j))))
    if not pairs:
        raise ValueError("Empty matching graph")
    return sorted(pairs)


def image_record(path):
    import cv2
    import numpy as np
    with Image.open(path) as original:
        original.load()
        raw_size = list(original.size)
        im = ImageOps.exif_transpose(original).convert("RGB")
    display_size = list(im.size)
    pixels = hashlib.sha256(f"RGB:{im.width}x{im.height}:".encode() + im.tobytes()).hexdigest()
    relative = path.relative_to(LIBRARY).as_posix()
    thumb = LIBRARY / "previews" / (hashlib.sha256(relative.encode()).hexdigest()[:20] + ".jpg")
    thumb.parent.mkdir(parents=True, exist_ok=True)
    preview = im.copy()
    preview.thumbnail((700, 450), Image.Resampling.LANCZOS)
    preview.save(thumb, quality=88)
    tiny = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = tiny[:, 1:] > tiny[:, :-1]
    dhash = sum(int(bit) << i for i, bit in enumerate(bits.flat))
    im.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    gray = np.asarray(im.convert("L"))
    keypoints, descriptors = cv2.ORB_create(nfeatures=3000).detectAndCompute(gray, None)
    points = np.array([k.pt for k in keypoints], dtype=np.float32)
    return {"file": relative, "preview": thumb.relative_to(LIBRARY).as_posix(),
            "sha256": file_hash(path), "pixel_sha256": pixels, "dhash": f"{dhash:016x}",
            "original_size": raw_size, "display_size": display_size,
            "bytes": path.stat().st_size}, (points, descriptors, gray.shape[::-1])


def geometry_check(a, b):
    import cv2
    import numpy as np
    points1, desc1, size1 = a
    points2, desc2, size2 = b
    base = {"method": "ORB-3000 / ratio-0.75 / RANSAC-5px at max side 1000", "status": "review", "matches": 0, "inliers": 0}
    if desc1 is None or desc2 is None or len(desc2) < 2:
        return base
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc1, desc2, k=2)
    good = [v[0] for v in matches if len(v) == 2 and v[0].distance < .75 * v[1].distance]
    base["matches"] = len(good)
    if len(good) < 8:
        return base
    p1 = np.float32([points1[m.queryIdx] for m in good])
    p2 = np.float32([points2[m.trainIdx] for m in good])
    cv2.setRNGSeed(20260909)
    matrix, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5)
    if matrix is None or mask is None:
        return base
    inside = mask.ravel().astype(bool)
    inliers = int(inside.sum())
    area = []
    for points, size in [(p1, size1), (p2, size2)]:
        spans = np.ptp(points[inside], axis=0)
        area.append(float(spans[0] * spans[1] / (size[0] * size[1])))
    base.update({"inliers": inliers, "inlier_ratio": round(inliers / len(good), 4), "inlier_bbox_fraction": area})
    if inliers >= 15 and inliers / len(good) >= .3 and min(area) >= .015:
        base["status"] = "supported"
    return base


def build_index():
    from .materials import get_manifest
    import cv2
    cv2.setNumThreads(1)
    status = read_json(PROVENANCE / "download-status.json")
    if status["failures"]:
        raise RuntimeError("Finish downloads before building the catalog.")
    records, features = {}, {}
    paths = sorted(p for p in ORIGINALS.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(image_record, p): p for p in paths}
        for n, future in enumerate(as_completed(futures), 1):
            record, feature = future.result()
            records[record["file"]], features[record["file"]] = record, feature
            if n % 100 == 0:
                print(f"Decoded/hash-checked images {n}/{len(paths)}", flush=True)
    sources, groups, candidates, graph_issues = [], [], [], []
    for sid, repo, prefix, depth, title in SPECS:
        meta = read_json(PROVENANCE / (repo.replace("/", "__") + ".json"))
        sources.append({"id": sid, "title": title, "author": repo.split("/")[0], "source_url": f"https://github.com/{repo}",
                        "repository": repo, "commit": meta["tree"]["sha"], "license": "未提供单独的数据许可证",
                        "license_evidence": f"provenance/{repo.replace('/', '__')}/README.md", "kind": "github_dataset"})
        folders = sorted((ORIGINALS / sid / prefix).iterdir())
        for folder in folders:
            if not folder.is_dir():
                continue
            # GES filenames are zero-padded; case-insensitive filename order matches Windows enumeration.
            photos = sorted((p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS), key=lambda p: p.name.casefold())
            if not photos:
                continue
            if sid == "ges":
                graph = folder / (folder.name + "-STITCH-GRAPH.txt")
                rejected = []
                pairs = graph_pairs(graph.read_text(encoding="utf-8-sig"), len(photos), rejected)
                graph_issues.extend({"source_id": sid, "scene": folder.name, **issue} for issue in rejected)
                pairing = "author_stitch_graph_filename_order"
            else:
                if len(photos) != 2:
                    raise ValueError(f"Expected two original inputs: {folder}")
                pairs, pairing = [(0, 1)], "author_two_image_folder"
            relative = [p.relative_to(LIBRARY).as_posix() for p in photos]
            groups.append({"source_id": sid, "scene": folder.name, "images": relative, "pairing": pairing, "edges": pairs})
            for i, j in pairs:
                candidates.append((sid, folder.name, relative[i], relative[j], pairing))
    sources.append({"id": "openpano", "title": "OpenPano · 全景照片序列", "author": "Yuxin Wu / ppwwyyxx",
                    "source_url": "https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1", "repository": "ppwwyyxx/OpenPano", "release": "0.1",
                    "license": "仓库代码 MIT；示例照片未单独声明", "license_evidence": "provenance/ppwwyyxx__OpenPano/LICENSE", "kind": "github_dataset"})
    folders = defaultdict(list)
    for path in paths:
        if path.is_relative_to(ORIGINALS / "openpano"):
            folders[path.parent].append(path)
    for folder, photos in sorted(folders.items()):
        photos.sort(key=lambda p: p.name.casefold())
        scene = folder.relative_to(ORIGINALS / "openpano").as_posix()
        relative = [p.relative_to(LIBRARY).as_posix() for p in photos]
        pairs = [(i, i + 1) for i in range(len(photos) - 1)]
        groups.append({"source_id": "openpano", "scene": scene, "images": relative, "pairing": "filename_adjacent_then_geometry_check", "edges": pairs})
        candidates.extend(("openpano", scene, relative[i], relative[j], "filename_adjacent_then_geometry_check") for i, j in pairs)
    cases, duplicates, omitted, by_pixels = [], [], [], {}
    for n, (sid, scene, left, right, pairing) in enumerate(candidates, 1):
        pixel_pair = tuple(sorted([records[left]["pixel_sha256"], records[right]["pixel_sha256"]]))
        provenance = {"source_id": sid, "scene": scene, "left": left, "right": right, "pairing": pairing}
        if pixel_pair[0] == pixel_pair[1]:
            omitted.append({**provenance, "reason": "identical_input_images"})
            continue
        if pixel_pair in by_pixels:
            duplicates.append({**provenance, "same_as": by_pixels[pixel_pair]["id"], "basis": "exact_decoded_pixels"})
            by_pixels[pixel_pair].setdefault("also_in", []).append(provenance)
            continue
        check = geometry_check(features[left], features[right])
        if sid == "openpano" and check["status"] != "supported":
            omitted.append({**provenance, "reason": "adjacent_filename_overlap_unconfirmed", "geometry_check": check})
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", scene.lower()).strip("-")
        suffix = hashlib.sha256((left + "|" + right).encode()).hexdigest()[:8]
        case_id = f"{sid}-{slug}-{suffix}"
        tags = {"spw": ["建筑结构"], "lpc": ["视差", "线条结构"], "rew": ["视差"],
                "ges": ["多视图", "建筑结构"], "openpano": ["全景序列"]}[sid].copy()
        for pattern, label in [("library|office|indoor|shelf|desk|cabinet|playroom|atrium|books|cup|computer", "室内线索"),
                               ("garden|park|lawn|forest|farmland|river|potberry|plantain", "植被线索")]:
            if re.search(pattern, scene, re.I):
                tags.append(label)
        case = {"id": case_id, "title": f"{sid.upper()} · {scene.split('/')[-1]} · {Path(left).stem} + {Path(right).stem}",
                **provenance, "kind": "real_capture", "reference": None, "split": "upstream_unspecified_local_evaluation",
                "preview_left": records[left]["preview"], "preview_right": records[right]["preview"],
                "tags": tags, "tag_basis": "论文研究重点与目录名线索；不是逐对人工难度标注",
                "geometry_check": check, "review_required": check["status"] != "supported", "prompt": REAL_PROMPT,
                "limitations": "没有完整真值，不能计算参考图 SSIM。原图未预对齐；网页显示缩略图，提交使用原图。"
                    + ("自动匹配支持不足，先目视确认共同视野；这不代表样例无效。" if check["status"] != "supported" else ""),
                "left_size": records[left]["original_size"], "right_size": records[right]["original_size"]}
        cases.append(case)
        by_pixels[pixel_pair] = case
        if n % 100 == 0:
            print(f"Checked pairs {n}/{len(candidates)}", flush=True)
    # Flag, rather than silently delete, resized/re-encoded lookalikes from other repositories.
    near = []
    for i, case in enumerate(cases):
        hashes = [int(records[case[k]]["dhash"], 16) for k in ("left", "right")]
        for other in cases[:i]:
            if case["source_id"] == other["source_id"]:
                continue
            old = [int(records[other[k]]["dhash"], 16) for k in ("left", "right")]
            distance = min(max((hashes[0] ^ old[0]).bit_count(), (hashes[1] ^ old[1]).bit_count()),
                           max((hashes[0] ^ old[1]).bit_count(), (hashes[1] ^ old[0]).bit_count()))
            if distance <= 6:
                case["possible_duplicate_of"] = other["id"]
                near.append({"case_id": case["id"], "possible_duplicate_of": other["id"], "dhash_max_distance": distance})
                break
    selected = recommend(cases)
    for source in sources:
        sid = source["id"]
        source["image_count"] = sum(k.startswith(f"originals/{sid}/") for k in records)
        source["case_count"] = sum(c["source_id"] == sid for c in cases)
        source["scene_count"] = sum(g["source_id"] == sid for g in groups)
    manifest = get_manifest()
    imported = {s["id"] for s in sources}
    manifest["sources"] = [s for s in manifest["sources"] if s["id"] not in imported] + sources
    manifest["cases"] = [c for c in manifest["cases"] if c["source_id"] not in imported] + cases
    manifest["version"] = 2
    manifest["evaluation_scope"] = "真实拍摄样例按作者双图目录、作者配对图或经过匹配检查的相邻帧组织；没有完整真值。受控裁剪样例保留完整参考图，但没有真实视差。参考图不会发送给模型。相同场景的多对照片不等于独立场景，跨仓库也存在重收录。"
    manifest["library_stats"] = {"downloaded_original_images": len(records), "unique_decoded_images": len({v["pixel_sha256"] for v in records.values()}),
        "original_bytes": sum(v["bytes"] for v in records.values()), "upstream_scene_folders": len(groups), "candidate_pairs": len(candidates),
        "real_pairs": len(cases), "exact_duplicate_pairs_removed": len(duplicates), "possible_near_duplicate_pairs": len(near),
        "omitted_candidates": len(omitted), "review_required_pairs": sum(c["review_required"] for c in cases), "recommended_real_pairs": len(selected)}
    write_json(PROVENANCE / "images.json", {"images": sorted(records.values(), key=lambda x: x["file"])})
    write_json(PROVENANCE / "sequences.json", {"groups": groups})
    write_json(PROVENANCE / "pair-audit.json", {"duplicates": duplicates, "possible_near_duplicates": near, "omitted_candidates": omitted, "upstream_graph_issues": graph_issues,
        "geometry_check_limits": "ORB/RANSAC is only an overlap sanity check; not a ground-truth alignment or stitching-quality score. Author pairs that fail are retained for review.",
        "ges_indexing": "Graph indices use case-insensitive filename order. Upstream Parameter.cpp uses directory enumeration without explicit sorting; filenames here are padded/consistent. Low-support edges remain flagged."})
    write_json(LIBRARY / "manifest.json", manifest)
    write_json(LIBRARY / "smoke-suite.json", {"name": f"第一轮推荐：{len(selected)} 对真实照片", "case_ids": [c["id"] for c in selected],
        "metrics": ["接缝是否自然", "物体数量是否保持", "文字是否准确", "直线结构是否保持", "是否编造纹理", "耗时与实际输出规格"], "paid_generation_run": False})
    contact_sheet(selected)
    print(json.dumps(manifest["library_stats"], ensure_ascii=False, indent=2), flush=True)


def recommend(cases):
    """Choose a small suite across sources and scenes, without using generated results."""
    priorities = {"spw": [], "lpc": ["cup", "library", "DFW_shelf", "school", "Potberry"],
                  "rew": ["railtracks", "intersection", "guardbar", "worktable", "wall"],
                  "ges": ["atrium", "times_square", "WandaIndoor", "Garden", "RailStation"],
                  "openpano": ["myself", "uav", "NSH", "CMU0", "flower"]}
    for c in cases:
        c.pop("recommended", None)
    selected = []
    for sid, keywords in priorities.items():
        available = [c for c in cases if c["source_id"] == sid and not c["review_required"] and not c.get("possible_duplicate_of")]
        scene_used, chosen = set(), []
        for keyword in keywords:
            c = next((c for c in available if keyword.lower() in c["scene"].lower() and c["scene"] not in scene_used), None)
            if c:
                chosen.append(c); scene_used.add(c["scene"])
        for c in available:
            if len(chosen) >= 5:
                break
            if c["scene"] not in scene_used:
                chosen.append(c); scene_used.add(c["scene"])
        for c in chosen:
            c["recommended"] = True
        selected.extend(chosen)
    return selected


def contact_sheet(cases):
    width, cell_w, cell_h = 1600, 800, 240
    rows = (len(cases) + 1) // 2
    canvas = Image.new("RGB", (width, 70 + rows * cell_h), "#f5f6f2")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    draw.text((25, 18), f"曦源 · GitHub 拼接测试库 / 推荐 {len(cases)} 对真实照片", fill="#20322f", font=title_font)
    for i, case in enumerate(cases):
        x, y = (i % 2) * cell_w, 70 + (i // 2) * cell_h
        draw.text((x + 15, y + 5), case["title"][:65], fill="#20322f", font=font)
        for j, key in enumerate(["preview_left", "preview_right"]):
            im = Image.open(LIBRARY / case[key]).convert("RGB")
            im.thumbnail((375, 187), Image.Resampling.LANCZOS)
            canvas.paste(im, (x + 15 + j * 390 + (375 - im.width)//2, y + 38 + (187 - im.height)//2))
    canvas.save(LIBRARY / "recommended-contact-sheet.jpg", quality=90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "index"])
    args = parser.parse_args()
    download() if args.action == "download" else build_index()
