from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .config import load_config
from .image_io import read_image, write_image
from .pipeline import StitchPipeline
from .run_io import environment, load_run, save_failure, save_run, sha256, write_json


def masked_metrics(image, reference, mask):
    valid = mask > 0
    if not np.any(valid):
        raise ValueError("评价区域为空。")
    diff = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(diff[valid]**2))
    from skimage.metrics import structural_similarity
    _, ssim_map = structural_similarity(image, reference, channel_axis=2, data_range=255, full=True)
    return {
        "mae": float(np.mean(np.abs(diff[valid]))),
        "psnr_db": float(10 * np.log10(255**2 / mse)) if mse > 0 else None,
        "perfect_match": mse == 0, "ssim": float(np.mean(ssim_map[valid])),
        "evaluated_pixels": int(valid.sum()),
    }


class LPIPSEvaluator:
    def __init__(self):
        import lpips
        import torch
        self.torch = torch
        self.model = lpips.LPIPS(net="alex", spatial=True).eval()

    def evaluate(self, image, reference, mask):
        torch = self.torch
        x, y, w, h = cv2.boundingRect(mask)
        # Include enough context for AlexNet and keep identical crops across methods.
        x0, y0 = max(0, x-32), max(0, y-32)
        x1, y1 = min(image.shape[1], x+w+32), min(image.shape[0], y+h+32)
        def tensor(a):
            rgb = np.ascontiguousarray(a[y0:y1, x0:x1, ::-1].transpose(2, 0, 1))
            return torch.from_numpy(rgb).float().unsqueeze(0) / 127.5 - 1
        with torch.inference_mode():
            values = self.model(tensor(image), tensor(reference))[0, 0].numpy()
        local_mask = mask[y0:y1, x0:x1] > 0
        return float(values[local_mask].mean())


def make_comparison(output: Path, images: dict, bbox):
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0-48), max(0, y0-48)
    rows = []
    for name, image in images.items():
        crop = image[y0:min(image.shape[0], y1+48), x0:min(image.shape[1], x1+48)]
        scale = min(340 / crop.shape[1], 340 / crop.shape[0])
        thumb = cv2.resize(crop, None, fx=scale, fy=scale)
        panel = np.full((390, 360, 3), 245, dtype=np.uint8)
        cv2.putText(panel, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        panel[44:44+thumb.shape[0], 10:10+thumb.shape[1]] = thumb
        rows.append(panel)
    write_image(output, np.concatenate(rows, axis=1))


def _safe_input(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("数据清单路径超出数据集目录。")
    return path


def run_benchmark(manifest_path: Path, output: Path, config: dict, *,
                  use_ai=False, limit=None, lpips_enabled=False, selected_ids=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if selected_ids:
        cases = [c for c in cases if c["id"] in selected_ids]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("没有选中的测试案例。")
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录必须为空，避免旧结果混入新实验。")
    output.mkdir(parents=True, exist_ok=True)
    root = manifest_path.parent
    evaluator = LPIPSEvaluator() if lpips_enabled else None
    pipeline = StitchPipeline(config)
    records, rows = [], []
    started = time.perf_counter()
    for i, case in enumerate(cases):
        case_id = case["id"]
        if not case_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in case_id):
            raise ValueError("案例 id 只能含英文字母、数字、下划线和连字符。")
        folder = output / case_id
        print(f"[{i+1}/{len(cases)}] {case_id}", flush=True)
        record = {"id": case_id, "category": case.get("category"), "source_id": case.get("source_id"),
                  "kind": case.get("kind", "user_capture"), "ai_status": "not_requested", "status": "failed"}
        try:
            a = _safe_input(root, case["image_a"])
            b = _safe_input(root, case["image_b"])
            record["input_sha256"] = {"a": sha256(a), "b": sha256(b)}
            expected = case.get("input_sha256")
            if expected and expected != record["input_sha256"]:
                raise ValueError("输入图片校验失败，文件与数据清单不一致。")
            if case.get("prepared_run"):
                prepared, _ = load_run(_safe_input(root,case["prepared_run"]))
            else:
                prepared = pipeline.run(a, b)
            save_run(folder, prepared, config, extra={"case": case})
            result = prepared
            if use_ai:
                try:
                    custom = read_image(_safe_input(root, case["mask"]))[:, :, 0] if case.get("mask") else None
                    result = pipeline.run(a, b, prepared=prepared, use_ai=True, custom_mask=custom)
                    if result.metrics.get("test_double"):
                        raise ValueError("实验禁止使用模拟 AI 输出。")
                    save_run(folder, result, config, extra={"case": case})
                    record["ai_status"] = "success"
                except Exception as exc:
                    record["ai_status"] = "failed"
                    record["ai_error"] = f"{type(exc).__name__}: {exc}"
                    save_failure(folder, exc, config, extra={"stage": "ai", "case": case})
            images = {"Feather": prepared.traditional_image, "Poisson": prepared.poisson_image}
            if record["ai_status"] == "success":
                images["AI"] = result.final_image
            record["timings"] = result.metrics
            if case.get("reference"):
                reference = read_image(_safe_input(root, case["reference"]))
                reg = result.registration
                h, w = result.traditional_image.shape[:2]
                transform = reg.canvas_transform @ np.asarray(case["reference_to_a"], dtype=float)
                ref_canvas = cv2.warpPerspective(reference, transform, (w, h))
                valid = cv2.warpPerspective(np.full(reference.shape[:2], 255, np.uint8), transform, (w, h))
                valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
                mask = cv2.bitwise_and(result.mask.binary_mask, valid)
                # Evaluate all methods in the same geometry and mask.
                record["evaluation_region"] = "hard seam mask intersected with reference validity eroded by 3 pixels"
                record["reference_sha256"] = sha256(_safe_input(root, case["reference"]))
                write_image(folder / "reference_canvas.png", ref_canvas)
                write_image(folder / "evaluation_mask.png", mask)
                record["quality"] = {}
                for method, image in images.items():
                    metrics = masked_metrics(image, ref_canvas, mask)
                    if evaluator:
                        metrics["lpips_alex_spatial"] = evaluator.evaluate(image, ref_canvas, mask)
                    record["quality"][method] = metrics
                    rows.append({"case_id": case_id, "source_id": case.get("source_id"),
                                 "category": case.get("category"), "method": method,
                                 "ai_status": record["ai_status"], **metrics})
                images = {"Reference": ref_canvas, **images}
            else:
                record["quality"] = None
                record["evaluation_note"] = "No aligned reference supplied; reference metrics are not computed."
            make_comparison(folder / "comparison.png", images, result.mask.bbox)
            record["status"] = "success" if record["ai_status"] != "failed" else "ai_failed"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            save_failure(folder, exc, config, extra={"case": case})
        records.append(record)
        write_json(folder / "evaluation.json", record)
        write_json(output / "progress.json", {"completed": len(records), "total": len(cases), "records": records})
    summary = {
        "schema_version": 1, "manifest_sha256": sha256(manifest_path), "config": deepcopy(config),
        "environment": environment(), "cases_requested": len(cases),
        "independent_source_count": len({c.get("source_id", c["id"]) for c in cases}),
        "registration_successes": sum(r["status"] != "failed" for r in records),
        "ai_requested": use_ai, "ai_successes": sum(r["ai_status"] == "success" for r in records),
        "ai_failures": sum(r["ai_status"] == "failed" for r in records),
        "lpips_requested": lpips_enabled, "elapsed_seconds": time.perf_counter()-started,
        "dataset_description": manifest.get("description"), "records": records,
        "conclusion": "Results describe this dataset only. No unmeasured AI quality or GPU performance is claimed.",
    }
    write_json(output / "summary.json", summary)
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    sections = []
    for r in records:
        label = html.escape(r["id"])
        error = html.escape(r.get("error") or r.get("ai_error") or "")
        photo = f'<a href="{label}/run.json">运行记录</a><br><img src="{label}/comparison.png" loading="lazy">' if r["status"] != "failed" else ""
        sections.append(f'<section><h2>{label}</h2><p>状态：{r["status"]} · AI：{r["ai_status"]}</p><p>{error}</p>{photo}</section>')
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>曦源实验结果</title><style>body{font:16px system-ui;background:#f4f5f7;color:#202a35;max-width:1500px;margin:30px auto;padding:20px}section{background:white;border-radius:12px;padding:20px;margin:18px 0}img{max-width:100%;height:auto}a{color:#176c79}</style><h1>曦源实验结果</h1>'
    document += f'<p>{len(cases)} 组案例，{summary["independent_source_count"]} 个独立来源。AI 成功 {summary["ai_successes"]} 组。</p>'
    document += '<p>'+html.escape(str(manifest.get("description", "")))+'</p>'
    document += '<p>未运行的 AI 不记为成功；失败案例完整保留。指标见 metrics.csv，环境与参数见 summary.json。</p>'+''.join(sections)+'</html>'
    (output / "index.html").write_text(document, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="批量拼接实验与效果报告")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--ai", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", help="逗号分隔的案例 id")
    args = parser.parse_args()
    try:
        result = run_benchmark(args.manifest, args.output, load_config(args.config), use_ai=args.ai,
                               limit=args.limit, lpips_enabled=args.lpips,
                               selected_ids=args.ids.split(",") if args.ids else None)
        print(json.dumps({k: result[k] for k in ("cases_requested", "registration_successes", "ai_successes", "ai_failures")}))
        return 0 if result["registration_successes"] == result["cases_requested"] and not result["ai_failures"] else 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
