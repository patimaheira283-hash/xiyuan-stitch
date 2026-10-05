from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "data" / "gpt-fusion-library"
RUNS = ROOT / "outputs" / "gpt-fusion"
MODEL = "gpt-image-2.5-sunburst"
MAINLINE_MODEL = "gpt-6-astra"
PROMPT = """These are two overlapping photographs of the SAME scene, not separate scenes.
Image 1 is the left view; Image 2 is the right view. Produce ONE continuous photographic panorama covering the union of both views. Identify the shared overlap and include every shared object only once. Match the viewpoint and exposure as needed, and make the join visually seamless.
Preserve the original scene, object identities, object counts, relative positions, straight lines, texture patterns, and all existing text exactly. Preserve the outer content of both views. Do not invent or remove scene content, beautify, restyle, add labels, frames, borders, or make a side-by-side collage. Change as little as possible outside the join. Return only the final stitched image."""


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").resolve()


def artifact_root() -> Path:
    return codex_home() / "output" / "imagegen"


def helper_path() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return codex_home() / "skills" / "openai-codex-image-skills" / "bin" / ("codex-image-helper" + suffix)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(repr=False)
class Provider:
    id: str
    name: str
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)
    mainline_model: str = "gpt-5.5"

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "source": "CC Switch", "configured": bool(self.api_key), "mainline_model": self.mainline_model}


def load_provider(provider_id: str | None = None, db_path: Path | None = None) -> Provider:
    db = (db_path or Path.home() / ".cc-switch" / "cc-switch.db").resolve()
    if not db.is_file():
        raise ValueError("未找到 CC Switch 配置库。")
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
        if provider_id:
            rows = conn.execute("SELECT id,name,settings_config FROM providers WHERE app_type='codex' AND id=?", (provider_id,)).fetchall()
        else:
            rows = conn.execute("SELECT id,name,settings_config FROM providers WHERE app_type='codex' AND is_current=1").fetchall()
    if len(rows) != 1:
        raise ValueError("CC Switch 必须有且仅有一个当前 Codex Provider，或显式指定 Provider ID。")
    pid, name, raw = rows[0]
    cfg = json.loads(raw)
    config = tomllib.loads(cfg["config"])
    selected = config.get("model_provider")
    base = config.get("model_providers", {}).get(selected, {}).get("base_url", "")
    auth = cfg.get("auth", {})
    if isinstance(auth, str):
        auth = json.loads(auth)
    key = auth.get("OPENAI_API_KEY", "")
    parts = urlsplit(base)
    if parts.scheme not in ("https", "http") or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("当前 CC Switch Provider 的地址不受支持。")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("当前 CC Switch Provider 没有 API key。")
    return Provider(pid, name, base, key, config.get("model", "gpt-5.5"))


def sanitize(message: str, provider: Provider | None = None) -> str:
    if provider:
        message = message.replace(provider.api_key, "[REDACTED]").replace(provider.base_url, "[provider]")
        hostname = urlsplit(provider.base_url).hostname
        if hostname:
            message = message.replace(hostname, "[provider-host]")
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(r"https?://[^\s\"<>]+", "[provider]", message)
    return message[:1200]


def validate_image(path: Path) -> None:
    if path.stat().st_size > 30 * 1024 * 1024:
        raise ValueError("单张输入图片不能超过 30 MB。")
    with Image.open(path) as im:
        if im.width * im.height > 40_000_000:
            raise ValueError("输入图片不能超过 4000 万像素。")
        im.verify()


def evaluate(output: Path, truth: Path | None) -> dict:
    if truth is None:
        return {"reference_available": False, "note": "没有真实完整参考图，不计算参考图指标。"}
    import numpy as np
    from skimage.metrics import structural_similarity

    ref = Image.open(truth).convert("RGB")
    result = Image.open(output).convert("RGB")
    a = np.asarray(ref)
    b = np.asarray(result.resize(ref.size, Image.Resampling.LANCZOS))
    delta = np.abs(a.astype(float) - b.astype(float))
    w = a.shape[1]
    overlap = slice(round(w * .375), round(w * .625))
    outside = np.concatenate([delta[:, :overlap.start], delta[:, overlap.stop:]], axis=1)
    return {
        "reference_available": True,
        "ssim_resized": float(structural_similarity(a, b, channel_axis=2, data_range=255)),
        "mae_rgb_0_255": float(delta.mean()),
        "overlap_mae_rgb_0_255": float(delta[:, overlap].mean()),
        "outside_overlap_mae_rgb_0_255": float(outside.mean()),
        "output_aspect_ratio_error": abs(result.width / result.height / (ref.width / ref.height) - 1),
        "reference_size": list(ref.size),
        "note": "仅将输出缩放到参考图尺寸后比较；没有几何对齐、颜色校正或修改原始输出。指标包含构图漂移，不能单独代表接缝质量。",
    }


def run_fusion(left: Path, right: Path, *, case_id: str = "custom", prompt: str = PROMPT,
               model: str = MODEL, quality: str = "high", size: str = "1536x1024",
               truth: Path | None = None, provider_id: str | None = None,
               dry_run: bool = False, run_id: str | None = None, route: str = "responses",
               mainline_model: str = MAINLINE_MODEL) -> dict:
    if route not in ("responses", "images"):
        raise ValueError("不支持的调用通道。")
    if model not in ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"):
        raise ValueError("本实验只允许 GPT Image 2.5，避免静默换用其他模型。")
    if quality not in ("low", "medium", "high"):
        raise ValueError("当前图像 helper 支持 low / medium / high。")
    if size not in ("1536x1024", "1024x1024", "2048x1024"):
        raise ValueError("不支持的输出尺寸。")
    if not prompt.strip() or len(prompt) > 10000:
        raise ValueError("提示词须为 1 到 10000 个字符。")
    for path in (left, right):
        validate_image(path)
    if truth:
        validate_image(truth)
    provider = load_provider(provider_id)
    helper = helper_path()
    if route == "images" and not helper.is_file():
        raise ValueError("未安装 openai-codex-image-skills 图像 helper。")
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("无效运行 ID。")
    folder = RUNS / run_id
    folder.mkdir(parents=True, exist_ok=False)
    inputs = []
    for name, path in (("left", left), ("right", right)):
        dst = folder / (name + path.suffix.lower())
        shutil.copyfile(path, dst)
        inputs.append(dst)
    reference = None
    if truth:
        reference = folder / "reference.png"
        Image.open(truth).convert("RGB").save(reference)
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    artifact_root().mkdir(parents=True, exist_ok=True)
    output = artifact_root() / ("xiyuan-gpt-fusion-" + run_id + ".png")
    env = os.environ.copy()
    env["XIYUAN_IMAGE_BASE_URL"] = provider.base_url
    env["XIYUAN_IMAGE_API_KEY"] = provider.api_key
    cmd = [str(helper), "--prompt-file", str(folder / "prompt.txt"), "--model", model,
           "--quality", quality, "--size", size, "--out", str(output),
           "--base-url-env", "XIYUAN_IMAGE_BASE_URL", "--api-key-env", "XIYUAN_IMAGE_API_KEY",
           "--allow-provider-override", "--max-retries", "0", "--timeout", "300", "--quiet"]
    for path in inputs:
        cmd += ["--image", str(path)]
    if dry_run:
        cmd.append("--dry-run")
    metadata = {
        "id": run_id, "case_id": case_id, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider.public(), "model_requested": model, "model_returned": None,
        "backend_identity_verified": False, "quality": quality, "size_requested": size,
        "input_files": [p.name for p in inputs], "input_sha256": [file_hash(p) for p in inputs],
        "reference_file": "reference.png" if reference else None,
        "preprocessing": "none; source image bytes sent unchanged", "mask_sent": False,
        "reference_sent_to_model": False, "retry_count": 0, "dry_run": dry_run,
        "route": route, "mainline_model_requested": mainline_model if route == "responses" else None,
        "usage": None, "cost_usd": None,
        "accounting_note": "实际扣费以服务商账单为准。上游模型身份无法独立验证。",
    }
    write_json(folder / "run.json", metadata)
    start = time.perf_counter()
    try:
        if route == "responses":
            from .responses_client import build_request, generate
            payload = build_request(inputs[0], inputs[1], prompt, mainline_model, model, quality, size)
            preview = json.loads(json.dumps(payload))
            for item in preview["input"][0]["content"]:
                if item["type"] == "input_image":
                    item["image_url"] = "[local input image; base64 omitted]"
            write_json(folder / "request-preview.json", preview)
            if dry_run:
                metadata["status"] = "dry_run"
            else:
                response_meta = generate(provider, payload, output)
                metadata.update(response_meta)
                metadata["accounting_note"] = "usage 来自 Responses 返回值；第三方实际扣费须以服务商账单为准。上游身份无法独立验证。"
        else:
            completed = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=330, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            raw = (completed.stdout + "\n" + completed.stderr).strip()
        if dry_run and route == "images":
            payload, _ = json.JSONDecoder().raw_decode(completed.stdout.lstrip())
            if payload.get("endpoint") != "/images/edits" or len(payload.get("files", [])) != 2:
                raise ValueError("dry-run 未确认两个图片输入。")
            write_json(folder / "request-preview.json", payload)
            metadata["status"] = "dry_run"
        elif not dry_run and output.is_file():
            validate_image(output)
            with Image.open(output) as im:
                metadata["size_actual"] = list(im.size)
            shutil.copyfile(output, folder / "result.png")
            metadata.update(status="succeeded", artifact_path=output.as_posix(), output_file="result.png",
                            output_sha256=file_hash(output), size_matches_request=("x".join(map(str, metadata["size_actual"])) == size))
            metadata["quality_returned"] = [i.get("quality") for i in metadata.get("image_tool_metadata", [])]
            metadata["model_returned"] = next((i.get("model") for i in metadata.get("image_tool_metadata", []) if i.get("model")), None)
            metadata["warnings"] = []
            if not metadata["size_matches_request"]:
                metadata["warnings"].append("实际输出尺寸与请求不同，已保留原始返回图片。")
            if any(q and q != quality for q in metadata["quality_returned"]):
                metadata["warnings"].append("图像工具回报的质量档位与请求不同。")
            if not metadata["model_returned"]:
                metadata["warnings"].append("返回值没有声明图像模型名，不能核实是否使用了请求的 GPT Image 2.5。")
            for i, extra in enumerate(metadata.get("artifacts", [])[1:], start=2):
                shutil.copyfile(extra, folder / f"result-{i}.png")
            try:
                metadata["evaluation"] = evaluate(output, reference)
            except Exception as exc:
                metadata["evaluation_error"] = sanitize(str(exc), provider)
        elif not dry_run:
            raise RuntimeError(sanitize((raw if route == "images" else "") or "图像服务未返回可用图片。", provider))
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = sanitize(str(exc), provider)
    finally:
        metadata["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        write_json(folder / "run.json", metadata)
    return metadata


def list_runs() -> list[dict]:
    result = []
    for path in sorted(RUNS.glob("*/run.json"), reverse=True):
        try:
            result.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return result
