"""Use the user's Codex Responses endpoint with the built-in image tool.

The user explicitly selected this route; it is distinct from the Images API helper.
Secrets and image payloads are never logged. Streaming is consumed in the backend.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import time
from pathlib import Path

import requests


def build_masked_edit_request(images: list[Path], mask: Path, prompt: str, mainline_model: str,
                              image_model: str, quality: str, size: str) -> dict:
    """The alpha mask applies to the first image, per the Responses API contract."""
    from PIL import Image
    if not images:
        raise ValueError("Masked edit requires a base image")
    with Image.open(images[0]) as base, Image.open(mask) as m:
        if m.size != base.size or m.mode != "RGBA":
            raise ValueError("Mask must be RGBA and match the first image dimensions")
        if m.getchannel("A").getextrema() != (0, 255):
            raise ValueError("Mask must contain editable and protected pixels")
    def data_url(path):
        return "data:" + (mimetypes.guess_type(path.name)[0] or "image/png") + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return {"model": mainline_model,
            "instructions": "Call image_generation exactly once to EDIT the FIRST image using the supplied alpha mask. The other images are references only. Preserve the first image's exact canvas, viewpoint, scale and framing. Repair only its masked seam; never create a new panorama or recompose the scene. Return the edited image.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}] + [{"type": "input_image", "image_url": data_url(p)} for p in images]}],
            "tools": [{"type": "image_generation", "model": image_model, "action": "edit", "quality": quality,
                       "size": size, "output_format": "png", "input_image_mask": {"image_url": data_url(mask)}}],
            "tool_choice": "required", "stream": True, "store": False}


def build_request(left: Path, right: Path, prompt: str, mainline_model: str, image_model: str, quality: str, size: str) -> dict:
    content = [{"type": "input_text", "text": prompt}]
    for image in (left, right):
        mime = mimetypes.guess_type(image.name)[0] or "image/png"
        content.append({"type": "input_image", "image_url": "data:" + mime + ";base64," + base64.b64encode(image.read_bytes()).decode("ascii")})
    return {"model": mainline_model, "instructions": "Use the image_generation tool to produce exactly one stitched image from the two provided photographs. Return the image, not code or a verbal explanation.",
            "input": [{"role": "user", "content": content}], "tools": [{"type": "image_generation", "model": image_model,
            "action": "edit", "quality": quality, "size": size, "output_format": "png"}],
            "tool_choice": "required", "stream": True, "store": False}


def events(lines):
    """SSE supports multi-line data and ignores comments and event labels."""
    parts = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line:
            if parts:
                value = "\n".join(parts)
                parts = []
                if value != "[DONE]":
                    yield json.loads(value)
        elif line.startswith("data:"):
            parts.append(line[5:].lstrip())
    if parts and "\n".join(parts) != "[DONE]":
        yield json.loads("\n".join(parts))


def generate(provider, payload: dict, output: Path) -> dict:
    from .core import sanitize, validate_image

    base = provider.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    headers = {"Authorization": "Bearer " + provider.api_key, "Content-Type": "application/json", "Accept": "text/event-stream"}
    saved, hashes, texts = [], set(), []
    result = {"artifacts": saved, "usage": None, "response_model": None, "response_id": None, "image_tool_metadata": []}

    def consume_item(item):
        if item.get("type") == "image_generation_call":
            b64 = item.get("result")
            if isinstance(b64, str) and b64:
                binary = base64.b64decode(b64, validate=True)
                digest = hashlib.sha256(binary).hexdigest()
                if digest in hashes:
                    return
                path = output if not saved else output.with_stem(output.stem + f"-{len(saved)+1}")
                path.write_bytes(binary)
                validate_image(path)
                hashes.add(digest)
                saved.append(path.as_posix())
                result["image_tool_metadata"].append({k: item[k] for k in ("id", "status", "model", "size", "quality", "output_format", "revised_prompt") if k in item})
        elif item.get("type") == "message":
            texts.extend(c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text")

    def consume_response(response):
        result["usage"] = response.get("usage", result["usage"])
        result["response_model"] = response.get("model", result["response_model"])
        result["response_id"] = response.get("id", result["response_id"])
        for item in response.get("output", []):
            consume_item(item)
        if response.get("error"):
            raise RuntimeError(sanitize(json.dumps(response["error"], ensure_ascii=False), provider))

    start = time.monotonic()
    try:
        with requests.post(base + "/responses", headers=headers, json=payload, stream=True, timeout=(20, 360)) as response:
            result["http_status"] = response.status_code
            result["request_id"] = response.headers.get("x-request-id")
            if not response.ok:
                detail = response.raw.read(8192, decode_content=True).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status_code}: " + sanitize(detail, provider))
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                for event in events(response.iter_lines()):
                    if time.monotonic() - start > 600:
                        raise TimeoutError("请求超过 10 分钟，已停止等待；未自动重复调用。")
                    event_type = event.get("type", "")
                    if event_type == "response.output_item.done":
                        consume_item(event.get("item", {}))
                    elif event_type in ("response.completed", "response.failed", "response.incomplete"):
                        consume_response(event.get("response", {}))
                        result["completion_event"] = event_type
                    elif event_type == "error":
                        raise RuntimeError(sanitize(json.dumps(event, ensure_ascii=False), provider))
            else:
                consume_response(response.json())
        if not saved:
            detail = " ".join(texts)[:600]
            raise RuntimeError("Responses 请求没有返回 image_generation_call 的图片结果。" + (" 返回文本：" + sanitize(detail, provider) if detail else ""))
    except Exception as exc:
        if not saved:
            raise
        result["partial_error"] = sanitize(str(exc), provider)
    return result
