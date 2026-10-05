"""Expiring invitation gateway for a local MVP; no API credentials in this process."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import threading
import time
from urllib.parse import quote, unquote, urlsplit

import requests

from .core import ROOT, RUNS, write_json
from .server import Handler, LocalServer, STATIC

UPSTREAM = "http://127.0.0.1:18767"
COOKIE = "__Host-xiyuan-guest"
SHARES = RUNS / "shares"
PUBLIC_RUN_FIELDS = {"id", "case_id", "status", "created_at", "model_requested", "model_returned", "quality",
                     "quality_returned", "size_requested", "size_actual", "size_matches_request", "elapsed_seconds",
                     "reference_file", "reference_sent_to_model", "evaluation", "dry_run"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def local_request(method, path, **kwargs):
    # Only a fixed loopback origin; never accept an upstream URL from a visitor.
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method, UPSTREAM + path, timeout=(5, 60), allow_redirects=False, **kwargs)


class ShareState:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.submissions = threading.Lock()
        self.attempts = defaultdict(deque)

    def load(self):
        with self.lock:
            return read_json(self.path)

    def change(self, fn):
        with self.lock:
            data = read_json(self.path)
            result = fn(data)
            write_json(self.path, data)
            return result

    def allow_login(self, address):
        with self.lock:
            now = time.monotonic()
            if len(self.attempts) > 2048:
                self.attempts.clear()
            attempts = self.attempts[address]
            while attempts and attempts[0] < now - 60:
                attempts.popleft()
            if len(attempts) >= 12:
                return False
            attempts.append(now)
            return True


def visible_run(run):
    value = {k: v for k, v in run.items() if k in PUBLIC_RUN_FIELDS}
    value["provider"] = {"name": "同学试用"}
    if run.get("status") == "failed":
        value["error"] = "本次生成失败，请让实验发起人查看本机记录。"
    return value


class ShareHandler(Handler):
    def setup(self):
        super().setup()
        self.connection.settimeout(35)
        self.session_cookie = None

    def end_headers(self):
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Robots-Tag", "noindex, nofollow, noarchive")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if self.session_cookie:
            self.send_header("Set-Cookie", self.session_cookie)
        super().end_headers()

    def state(self):
        return self.server.share.load()

    def active(self):
        if not self.valid_host():
            self.send_json({"error": "访问地址无效。"}, 403)
            return None
        data = self.state()
        if not data.get("enabled") or time.time() >= data["expires_at"] or (self.server.share.path.parent / "STOP").exists():
            self.send_json({"error": "本次同学试用已结束，请联系发起人重新开放。"}, 410)
            return None
        return data

    def guest(self, data):
        try:
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            sid = cookies[COOKIE].value if COOKIE in cookies else ""
        except Exception:
            return None
        return sid if sid in data["sessions"] else None

    def allowed_runs(self, data, sid):
        return set(data["gallery_ids"]) | {rid for rid, owner in data["owners"].items() if owner == sid}

    def share_info(self, data):
        return {"expires_at": data["expires_at"], "remaining": max(0, data["max_generations"] - data["used_generations"]),
                "max_generations": data["max_generations"]}

    def do_GET(self):
        data = self.active()
        if data is None:
            return
        path = unquote(urlsplit(self.path).path)
        sid = self.guest(data)
        if path == "/join.js":
            self.send_data((STATIC / "join.js").read_bytes(), "text/javascript; charset=utf-8")
            return
        if path == "/" and sid is None:
            self.send_data((STATIC / "join.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/style.css" and sid is None:
            self.send_data((STATIC / "style.css").read_bytes(), "text/css; charset=utf-8")
            return
        if sid is None:
            self.send_json({"error": "请使用发起人提供的邀请链接进入。"}, 401)
            return
        allowed = self.allowed_runs(data, sid)
        try:
            if path in {"/api/state", "/api/runs"}:
                response = local_request("GET", path)
                response.raise_for_status()
                body = response.json()
                body["runs"] = [visible_run(r) for r in body["runs"] if r["id"] in allowed]
                body["share"] = self.share_info(data)
                if path == "/api/state":
                    body["provider"] = {"name": "同学试用", "configured": body["provider"].get("configured", False)}
                self.send_json(body)
                return
            if path.startswith("/runs/"):
                parts = path.split("/")
                if len(parts) != 4 or parts[2] not in allowed or parts[3] not in {"result.png", "left.png", "right.png", "reference.png", "left.jpg", "right.jpg"}:
                    self.send_json({"error": "没有这份结果的访问权限。"}, 403)
                    return
            elif path not in {"/", "/app.js", "/style.css"} and not path.startswith("/library/"):
                self.send_json({"error": "未找到页面。"}, 404)
                return
            response = local_request("GET", quote(path, safe="/"))
            if response.status_code != 200:
                self.send_json({"error": "暂时无法读取这份素材。"}, response.status_code)
            else:
                self.send_data(response.content, response.headers.get("Content-Type", "application/octet-stream"))
        except requests.RequestException:
            self.send_json({"error": "本机实验服务暂时不可用，请稍后重试。"}, 503)

    def do_POST(self):
        data = self.active()
        if data is None:
            return
        if not data.get("public_origin") or self.headers.get("Origin") != data["public_origin"]:
            self.send_json({"error": "请求来源校验失败。"}, 403)
            return
        path = urlsplit(self.path).path
        if path not in {"/api/join", "/api/run"}:
            self.send_json({"error": "未找到接口。"}, 404)
            return
        sid = self.guest(data)
        if path == "/api/run" and sid is None:
            self.send_json({"error": "试用身份已失效，请重新打开邀请链接。"}, 401)
            return
        if path == "/api/join" and not self.server.share.allow_login(self.headers.get("CF-Connecting-IP", self.client_address[0])):
            self.send_json({"error": "尝试过于频繁，请一分钟后重试。"}, 429)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            maximum = 2048 if path == "/api/join" else 85 * 1024 * 1024
            if not 0 < length <= maximum or self.headers.get("Transfer-Encoding"):
                raise ValueError("请求大小无效。")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("请求格式无效。")
        except (ValueError, TimeoutError):
            self.send_json({"error": "请求格式或大小无效。"}, 400)
            return
        if path == "/api/join":
            invite = body.get("invite", "")
            if not isinstance(invite, str) or not secrets.compare_digest(invite, data["invite"]):
                self.send_json({"error": "邀请码不正确或已经失效。"}, 401)
                return
            if len(data["sessions"]) >= 256 and not sid:
                self.send_json({"error": "本次试用人数已满，请联系发起人。"}, 429)
                return
            sid = sid or secrets.token_urlsafe(32)
            self.server.share.change(lambda value: value["sessions"].__setitem__(sid, {"joined_at": time.time()}))
            seconds = max(0, int(data["expires_at"] - time.time()))
            self.session_cookie = f"{COOKIE}={sid}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age={seconds}"
            self.send_json({"ok": True})
            return
        if not isinstance(body.get("dry_run", False), bool) or not isinstance(body.get("prompt", ""), str) or len(body.get("prompt", "")) > 8000:
            self.send_json({"error": "提示词或检查参数无效。"}, 400)
            return
        options = {"model": {"gpt-image-2.5-sunburst", "gpt-image-2.5-flare"}, "quality": {"high", "medium", "low"},
                   "size": {"1536x1024", "1024x1024", "2048x1024"}}
        if any(k in body and (not isinstance(body[k], str) or body[k] not in values) for k, values in options.items()):
            self.send_json({"error": "请选择页面提供的模型、质量和尺寸。"}, 400)
            return
        dry_run = body.get("dry_run", False)
        if not self.server.share.submissions.acquire(blocking=False):
            self.send_json({"error": "正在接收另一位同学的任务，请稍后重试。"}, 409)
            return
        reserved = False
        try:
            data = self.state()
            if time.time() >= data["expires_at"] or not data.get("enabled"):
                self.send_json({"error": "本次试用已结束。"}, 410)
                return
            if not dry_run and data["used_generations"] >= data["max_generations"]:
                self.send_json({"error": "本次共享生图额度已用完，仍可浏览素材和结果。"}, 429)
                return
            if not dry_run:
                # Reserve before contacting upstream; ambiguous timeouts never reopen a paid slot.
                self.server.share.change(lambda value: value.__setitem__("used_generations", value["used_generations"] + 1))
                reserved = True
            fields = {"case_id", "uploads", "prompt", "model", "quality", "size", "dry_run"}
            response = local_request("POST", "/api/run", json={k: v for k, v in body.items() if k in fields},
                                     headers={"Origin": UPSTREAM, "X-Fusion-Token": self.headers.get("X-Fusion-Token", "")})
            result = response.json()
            if response.status_code == 202:
                rid = result["id"]
                self.server.share.change(lambda value: value["owners"].__setitem__(rid, sid))
                self.send_json({"id": rid}, 202)
            else:
                if reserved:
                    self.server.share.change(lambda value: value.__setitem__("used_generations", value["used_generations"] - 1))
                message = "已有任务正在生成，请稍后重试。" if response.status_code == 409 else "请求未通过检查，请确认两张图片和所选素材。"
                self.send_json({"error": message}, response.status_code)
        except (requests.RequestException, ValueError, KeyError):
            self.send_json({"error": "本机服务响应异常，请勿重复提交，先联系发起人检查记录。"}, 503)
        finally:
            self.server.share.submissions.release()


def prepare(hours=24, limit=30, port=18768):
    SHARES.mkdir(parents=True, exist_ok=True)
    latest = SHARES / "latest.json"
    if latest.is_file():
        previous = read_json(latest)
        state_path = Path(previous["state_path"])
        if state_path.is_file():
            state = read_json(state_path)
            if state["enabled"] and state["expires_at"] > time.time():
                raise RuntimeError("已有未过期的共享入口；先查看或关闭，不创建重复隧道。")
    response = local_request("GET", "/api/state")
    response.raise_for_status()
    current = response.json()
    if current.get("app") != "gpt-fusion-mvp":
        raise RuntimeError("本机端口不是图像融合实验室。")
    folder = SHARES / datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / "state.json"
    known = {c["id"] for c in current["manifest"]["cases"]}
    data = {"enabled": True, "created_at": time.time(), "expires_at": time.time() + hours * 3600,
            "max_generations": limit, "used_generations": 0, "port": port, "public_origin": None,
            "invite": secrets.token_urlsafe(24), "sessions": {}, "owners": {},
            "gallery_ids": [r["id"] for r in current["runs"] if r.get("case_id") in known and r["status"] == "succeeded"]}
    write_json(path, data)
    write_json(latest, {"state_path": str(path)})
    return path


def host(path):
    share = ShareState(path)
    state = share.load()
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        raise RuntimeError("未安装 cloudflared。")
    server = LocalServer(("127.0.0.1", state["port"]), ShareHandler)
    server.share = share
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log = (path.parent / "tunnel.log").open("w", encoding="utf-8")
    child = subprocess.Popen([cloudflared, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{state['port']}",
                              "--http-host-header", f"127.0.0.1:{state['port']}", "--protocol", "http2"],
                             stdout=log, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    write_json(path.parent / "processes.json", {"host_pid": os.getpid(), "cloudflared_pid": child.pid, "state_path": str(path)})
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and child.poll() is None:
            text = (path.parent / "tunnel.log").read_text(encoding="utf-8", errors="replace")
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
            if match:
                origin = match[0]
                share.change(lambda value: value.__setitem__("public_origin", origin))
                print(json.dumps({"url": origin + "/#invite=" + state["invite"], "expires_at": state["expires_at"]}), flush=True)
                break
            time.sleep(1)
        else:
            raise RuntimeError("临时隧道未能启动，请查看本地 tunnel.log。")
        while child.poll() is None:
            state = share.load()
            if not state["enabled"] or time.time() >= state["expires_at"] or (path.parent / "STOP").exists():
                break
            time.sleep(2)
    finally:
        share.change(lambda value: value.__setitem__("enabled", False))
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        server.shutdown()
        server.server_close()
        log.close()


def stop():
    path = Path(read_json(SHARES / "latest.json")["state_path"])
    (path.parent / "STOP").touch()
    print("临时共享已关闭，隧道会在数秒内退出。本机实验页继续运行。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "host", "stop", "status"])
    parser.add_argument("--state", type=Path)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.hours, args.limit))
    elif args.action == "host":
        host(args.state)
    elif args.action == "stop":
        stop()
    else:
        value = read_json(Path(read_json(SHARES / "latest.json")["state_path"]))
        print(json.dumps({"enabled": value["enabled"], "url": (value.get("public_origin") or "") + "/#invite=" + value["invite"],
                          "remaining": value["max_generations"] - value["used_generations"],
                          "expires": datetime.fromtimestamp(value["expires_at"]).astimezone().isoformat()}, ensure_ascii=False, indent=2))
