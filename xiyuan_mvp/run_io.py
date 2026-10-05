from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from .image_io import read_image, write_image
from .types import MaskResult, RegistrationResult, StitchResult


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment() -> dict:
    packages = {}
    for name in ("xiyuan-stitch", "numpy", "opencv-python", "Pillow", "PyYAML", "PySide6",
                 "torch", "torchvision", "diffusers", "transformers", "accelerate", "lpips"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def write_json(path: str | Path, data: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def save_run(output: str | Path, result: StitchResult, config: dict, *, extra: dict | None = None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reg = result.registration
    images = {
        "input_a": reg.image_a, "input_b": reg.image_b,
        "01_matches": reg.match_image, "02_warped_a": reg.canvas_a,
        "03_warped_b": reg.canvas_b, "04_overlap_mask": reg.overlap_mask,
        "05_seam_mask": result.mask.binary_mask, "05_soft_mask": result.mask.soft_mask,
        "06_traditional": result.traditional_image, "mask_a": reg.mask_a, "mask_b": reg.mask_b,
    }
    if result.poisson_image is not None:
        images["06_poisson"] = result.poisson_image
    if result.repair_region is not None:
        images["repair_region"] = result.repair_region
    if result.repair_base_image is not None:
        images["06_initial"] = result.repair_base_image
    for index, source in enumerate(result.source_images,1):
        images[f"sequence_{index:03d}"] = source
    if result.metrics.get("use_ai"):
        images["07_ai_result"] = result.final_image
    else:
        # A previously generated image must not masquerade as this run's output.
        (output / "07_ai_result.png").unlink(missing_ok=True)
    artifacts = {}
    for name, image in images.items():
        filename = name + ".png"
        write_image(output / filename, image)
        artifacts[name] = {"file": filename, "sha256": sha256(output / filename)}
    report = {
        "schema_version": 1, "status": "success", "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": result.metrics, "config": config, "environment": environment(),
        "homography_b_to_a": reg.homography_b_to_a.tolist(),
        "canvas_transform": reg.canvas_transform.tolist(), "mask_bbox": list(result.mask.bbox), "mask_method": result.mask.method,
        "artifacts": artifacts, "extra": extra or {},
    }
    write_json(output / "run.json", report)
    return report


def save_failure(output: str | Path, exc: Exception, config: dict, *, extra: dict | None = None) -> None:
    write_json(Path(output) / "error.json", {
        "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "environment": environment(), "extra": extra or {},
    })


def load_run(path: str | Path) -> tuple[StitchResult, dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "run.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "success" or report.get("schema_version") != 1:
        raise ValueError("只支持本程序保存的成功运行记录。")
    root = path.parent.resolve()
    def get(name, gray=False):
        artifact = report["artifacts"][name]
        item = (root / artifact["file"]).resolve()
        if not item.is_relative_to(root) or sha256(item) != artifact["sha256"]:
            raise ValueError(f"结果文件路径或校验值异常：{name}")
        image = read_image(item)
        return image[:, :, 0] if gray else image
    reg = RegistrationResult(
        get("input_a"), get("input_b"), get("02_warped_a"), get("03_warped_b"),
        get("mask_a", True), get("mask_b", True), get("04_overlap_mask", True),
        np.asarray(report["homography_b_to_a"]), np.asarray(report["canvas_transform"]),
        get("01_matches"), report["metrics"],
    )
    mask = MaskResult(get("05_seam_mask", True), get("05_soft_mask", True), tuple(report["mask_bbox"]), report.get("mask_method","center"))
    traditional = get("06_traditional")
    initial = get("06_initial") if "06_initial" in report["artifacts"] else traditional.copy()
    final = get("07_ai_result") if report["metrics"]["use_ai"] else initial.copy()
    poisson = get("06_poisson") if "06_poisson" in report["artifacts"] else None
    region = get("repair_region",True) if "repair_region" in report["artifacts"] else reg.overlap_mask
    sources = [get(name) for name in sorted(report["artifacts"]) if name.startswith("sequence_")]
    return StitchResult(traditional, final, reg, mask, report["metrics"], poisson, region, sources, initial), report
