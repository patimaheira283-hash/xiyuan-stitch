from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import secrets
import socket
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageOps

from .core import LIBRARY, MODEL, MAINLINE_MODEL, PROMPT, ROOT, RUNS, list_runs, load_provider, run_fusion
from .materials import get_case, get_manifest

TOKEN = secrets.token_urlsafe(32)
BUSY = threading.Lock()
STATIC = Path(__file__).parent / "web"
UPLOADS = RUNS / "uploads"
ALLOWED_ASSETS = {"left.png", "right.png", "left.jpg", "right.jpg", "left.jpeg", "right.jpeg", "reference.png", "result.png", "prompt.txt", "run.json"}


def safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("无效文件路径。")
    return path


def public_run(run: dict) -> dict:
    value = dict(run)
    value.pop("artifact_path", None)
    return value


def save_upload(value: str, folder: Path, name: str) -> Path:
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("图片上传内容无效。") from exc
    if len(raw) > 30 * 1024 * 1024:
        raise ValueError("单张图片不能超过 30 MB。")
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError("图片不能超过 4000 万像素。")
        image = ImageOps.exif_transpose(image).convert("RGB")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        image.save(path)
    return path


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_data(self, data: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: dict | list, status: int = 200):
        self.send_data(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

    def valid_host(self) -> bool:
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self.valid_host():
            self.send_json({"error": "只允许本机访问。"}, 403)
            return
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/state":
                try:
                    provider = load_provider().public()
                except Exception as exc:
                    provider = {"configured": False, "error": str(exc)}
                self.send_json({"app": "gpt-fusion-mvp", "token": TOKEN, "provider": provider, "model": MODEL, "mainline_model": MAINLINE_MODEL, "prompt": PROMPT,
                                "manifest": get_manifest(), "runs": [public_run(r) for r in list_runs()], "busy": BUSY.locked()})
                return
            if path == "/api/runs":
                self.send_json({"runs": [public_run(r) for r in list_runs()], "busy": BUSY.locked()})
                return
            if path in ("/", "/app.js", "/style.css"):
                file = STATIC / ("index.html" if path == "/" else path[1:])
            elif path.startswith("/library/"):
                relative = path[len("/library/"):]
                manifest = get_manifest()
                known = {c[k] for c in manifest["cases"] for k in ("left", "right", "reference", "preview_left", "preview_right") if c.get(k)}
                known.update(s["file"] for s in manifest["sources"] if s.get("file"))
                if relative not in known:
                    raise ValueError("无效素材文件。")
                file = safe_child(LIBRARY, relative)
            elif path.startswith("/runs/"):
                relative = path[len("/runs/"):]
                if len(Path(relative).parts) != 2 or Path(relative).name not in ALLOWED_ASSETS:
                    raise ValueError("无效结果文件。")
                file = safe_child(RUNS, relative)
            else:
                self.send_json({"error": "未找到页面。"}, 404)
                return
            if not file.is_file():
                self.send_json({"error": "文件尚未生成。"}, 404)
                return
            content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            if file.suffix in (".html", ".css", ".js", ".txt"):
                content_type += "; charset=utf-8"
            self.send_data(file.read_bytes(), content_type)
        except (ValueError, OSError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def do_POST(self):
        origin = self.headers.get("Origin")
        allowed_origins = (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}")
        if not self.valid_host() or self.headers.get("X-Fusion-Token") != TOKEN or (origin and origin not in allowed_origins):
            self.send_json({"error": "请求来源校验失败。"}, 403)
            return
        if self.path != "/api/run":
            self.send_json({"error": "无效接口。"}, 404)
            return
        if not BUSY.acquire(blocking=False):
            self.send_json({"error": "已有任务正在运行，请等待完成。"}, 409)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 85 * 1024 * 1024:
                raise ValueError("请求大小无效。")
            data = json.loads(self.rfile.read(length))
            rid = "web-" + uuid.uuid4().hex[:16]
            truth = None
            default_prompt = PROMPT
            if data.get("uploads"):
                uploads = data["uploads"]
                if not isinstance(uploads, list) or len(uploads) != 2:
                    raise ValueError("请提供两张图片。")
                left = save_upload(uploads[0], UPLOADS / rid, "left.png")
                right = save_upload(uploads[1], UPLOADS / rid, "right.png")
                cid = "custom"
            else:
                case = get_case(data.get("case_id", ""))
                left, right = LIBRARY / case["left"], LIBRARY / case["right"]
                truth, cid = LIBRARY / case["reference"] if case.get("reference") else None, case["id"]
                default_prompt = case.get("prompt", PROMPT)
            kwargs = {"case_id": cid, "truth": truth, "prompt": data.get("prompt", default_prompt),
                      "model": data.get("model", MODEL), "quality": data.get("quality", "high"),
                      "size": data.get("size", "1536x1024"), "dry_run": bool(data.get("dry_run")), "run_id": rid}

            def work():
                try:
                    run_fusion(left, right, **kwargs)
                except Exception as exc:
                    from .core import write_json, sanitize
                    write_json(RUNS / rid / "run.json", {"id": rid, "case_id": cid, "status": "failed", "error": sanitize(str(exc)), "dry_run": kwargs["dry_run"]})
                finally:
                    BUSY.release()

            threading.Thread(target=work, daemon=True).start()
            self.send_json({"id": rid}, 202)
        except Exception as exc:
            BUSY.release()
            self.send_json({"error": str(exc)}, 400)


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(port: int = 18767, open_browser: bool = False):
    get_manifest()
    url = f"http://127.0.0.1:{port}"
    try:
        server = LocalServer(("127.0.0.1", port), Handler)
    except OSError:
        if open_browser:
            import requests
            try:
                existing = requests.get(url + "/api/state", timeout=2).json()
                if existing.get("app") == "gpt-fusion-mvp":
                    webbrowser.open(url)
                    return
            except (ValueError, requests.RequestException):
                pass
        raise RuntimeError(f"端口 {port} 已被占用，请用 --port 指定其他端口。") from None
    print(f"GPT 融合实验室：{url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
