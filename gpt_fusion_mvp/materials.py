"""Build controlled overlapping pairs from attributed, locally packaged sources."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from .core import LIBRARY, file_hash, write_json


def build_library() -> dict:
    import skimage
    from skimage import data

    sources_dir = LIBRARY / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    packaged = Path(skimage.__file__).parent / "data"
    definitions = [
        ("brick", "砖墙", "brick.png", "CC0-1.0", "CC0Textures", "https://cc0textures.com/view.php?tex=Bricks25", data.brick),
        ("coffee", "咖啡与木桌", "coffee.png", "CC0-1.0", "Rachel Michetti", "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee", data.coffee),
        ("rocket", "火箭与天空", "rocket.jpg", "Public domain", "SpaceX", "https://www.flickr.com/photos/spacexphotos/16511594820/", data.rocket),
        ("grass", "草地", "grass.png", "CC0-1.0", "linolafett", "https://www.deviantart.com/linolafett/art/Grass-01-434853879", data.grass),
    ]
    sources = []
    images = []
    for sid, title, filename, license_name, author, url, loader in definitions:
        original = packaged / filename
        if not original.is_file():
            raise FileNotFoundError(f"本地 scikit-image 素材不存在：{filename}")
        im = Image.open(original).convert("RGB")
        dst = sources_dir / (sid + ".png")
        im.save(dst)
        sources.append({"id": sid, "title": title, "file": f"sources/{sid}.png", "license": license_name,
                        "author": author, "source_url": url, "packaged_by": f"scikit-image {skimage.__version__}",
                        "license_evidence": loader.__doc__, "sha256": file_hash(dst), "original_size": list(im.size)})
        images.append((sid, title, im))
    chart = Image.new("RGB", (1536, 1024), "#f7f6f1")
    draw = ImageDraw.Draw(chart)
    fontpath = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(fontpath), 44) if fontpath.is_file() else ImageFont.load_default(size=44)
    small = ImageFont.truetype(str(fontpath), 30) if fontpath.is_file() else ImageFont.load_default(size=30)
    draw.text((65, 55), "XIYUAN / FUSION TEST / 2026", font=font, fill="#182d38")
    for row in range(4):
        for col in range(6):
            x, y = 60 + col * 235, 160 + row * 180
            draw.rectangle((x, y, x+210, y+145), outline="#263c46", width=3)
            draw.text((x+15, y+16), f"R{row+1} C{col+1}", font=small, fill="#182d38")
            draw.text((x+15, y+70), f"{(row+1)*100+col+1:03d}", font=font, fill=["#b24a38", "#277765", "#2a5e92"][col%3])
    draw.text((65, 930), "LEFT EDGE  |  Preserve every number and line  |  RIGHT EDGE", font=small, fill="#182d38")
    chart_path = sources_dir / "text-grid.png"
    chart.save(chart_path)
    sources.append({"id": "text-grid", "title": "文字与几何校验板", "file": "sources/text-grid.png", "license": "CC0-1.0",
                    "author": "Project-authored deterministic test chart", "source_url": None, "sha256": file_hash(chart_path),
                    "original_size": list(chart.size), "license_evidence": "项目程序绘制的测试板，以 CC0 提供；不是 AI 生成的自然照片。"})
    images.append(("text-grid", "文字与几何校验板", chart))
    cases = []
    for sid, title, source in images:
        # Native-size center crop to 3:2; never upscale the photographic sources.
        w, h = source.size
        if w / h > 1.5:
            nw = int(h * 1.5)
            reference = source.crop(((w-nw)//2, 0, (w-nw)//2+nw, h))
        else:
            nh = int(w / 1.5)
            reference = source.crop((0, (h-nh)//2, w, (h-nh)//2+nh))
        w, h = reference.size
        end, start = round(w * .625), round(w * .375)
        for variant in ("clean", "exposure"):
            cid = f"{sid}-{variant}"
            folder = LIBRARY / cid
            folder.mkdir(exist_ok=True)
            left = reference.crop((0, 0, end, h))
            right = reference.crop((start, 0, w, h))
            if variant == "exposure":
                right = ImageEnhance.Brightness(right).enhance(1.18)
            left.save(folder / "left.png")
            right.save(folder / "right.png")
            reference.save(folder / "reference.png")
            cases.append({"id": cid, "title": title + (" · 原始曝光" if variant == "clean" else " · 右图增亮 18%"),
                          "source_id": sid, "kind": "controlled_crop", "left": f"{cid}/left.png", "right": f"{cid}/right.png",
                          "reference": f"{cid}/reference.png", "reference_size": list(reference.size),
                          "left_box": [0, 0, end, h], "right_box": [start, 0, w, h],
                          "overlap_fraction_of_union": (end-start)/w,
                          "right_brightness_factor": 1.18 if variant == "exposure" else 1,
                          "limitations": "同一完整图裁出两张重叠图；不包含真实相机位移、视差或动态物体。参考图仅用于评估，绝不发送给模型。"})
    previous_path = LIBRARY / "manifest.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    source_ids, case_ids = {s["id"] for s in sources}, {c["id"] for c in cases}
    sources.extend(s for s in previous.get("sources", []) if s["id"] not in source_ids)
    cases.extend(c for c in previous.get("cases", []) if c["id"] not in case_ids)
    manifest = {**previous, "version": max(1, previous.get("version", 1)), "sources": sources, "cases": cases,
                "evaluation_scope": "首轮验证直接多图融合与内容保真；不能据此证明真实大视差拼接能力。"}
    if previous.get("version", 1) >= 2:
        manifest["evaluation_scope"] = previous["evaluation_scope"]
    write_json(LIBRARY / "manifest.json", manifest)
    return manifest


def get_manifest() -> dict:
    path = LIBRARY / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else build_library()


def get_case(case_id: str) -> dict:
    for case in get_manifest()["cases"]:
        if case["id"] == case_id:
            return case
    raise ValueError("没有这个测试素材。")
