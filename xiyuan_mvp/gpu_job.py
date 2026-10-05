"""Run a desktop-exported job without changing prepared geometry or painted masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .config import load_config
from .image_io import read_image
from .pipeline import StitchPipeline
from .run_io import load_run, save_failure, save_run, sha256, write_json


def safe_extract(archive_path, destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) > 1000 or sum(i.file_size for i in entries) > 2 * 1024**3:
            raise ValueError("任务 ZIP 超过 1000 文件或 2 GiB 解压上限。")
        for item in entries:
            name = item.filename.replace("\\", "/")
            target = (root / name).resolve()
            if ":" in name or not target.is_relative_to(root) or ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("任务 ZIP 包含非法路径。")
        archive.extractall(root)


def run_job(job, output, *, use_ai=True, pipeline_factory=StitchPipeline):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("请选择空的结果目录，避免覆盖已有实验。")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    try:
        with tempfile.TemporaryDirectory(prefix="xiyuan-job-") as temporary:
            root = Path(temporary)
            safe_extract(job, root)
            config = load_config(root / "config.yaml")
            if use_ai:
                config["inpainting"].update(device="cuda", allow_cpu=False, local_files_only=False,
                                           cache_dir=str(Path.home() / ".cache/xiyuan-models"))
                if config['inpainting'].get('engine') in ('neural_alignment','hybrid'):
                    config['neural_alignment']['device'] = 'cuda'
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1 or len(manifest.get("cases", [])) != 1:
                raise ValueError("需要桌面端导出的单任务 ZIP（可包含多图拼接）。")
            case = manifest["cases"][0]
            def member(key):
                path = (root / case[key]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("任务引用了 ZIP 外的文件。")
                return path
            prepared, _ = load_run(member("prepared_run"))
            painted = read_image(member("mask"))[:, :, 0]
            pipe = pipeline_factory(config)
            result = pipe.run(prepared.registration.image_a, prepared.registration.image_b,
                              prepared=prepared, custom_mask=painted, use_ai=use_ai)
            base = prepared.repair_base_image if prepared.repair_base_image is not None else prepared.traditional_image
            outside = result.mask.soft_mask == 0
            if not np.array_equal(result.final_image[outside], base[outside]):
                raise AssertionError("修复区域外的像素发生变化。")
            save_run(output, result, config, extra={"input_job_sha256": sha256(job),
                     "geometry_preserved": True, "outside_mask_unchanged": True})
            return result
    except Exception as exc:
        save_failure(output, exc, config)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="验证任务回传，不运行 AI")
    args = parser.parse_args()
    run_job(args.job, args.output, use_ai=not args.validate_only)


if __name__ == "__main__":
    main()
