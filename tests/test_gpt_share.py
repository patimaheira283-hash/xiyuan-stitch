import json
import threading
import time
from http.cookies import SimpleCookie

import pytest
import requests

from gpt_fusion_mvp import share
from gpt_fusion_mvp.core import write_json


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    write_json(path, {"enabled": True, "expires_at": time.time()+3600, "public_origin": "https://class.example",
                     "invite": "test-invitation", "sessions": {}, "owners": {}, "gallery_ids": ["demo"],
                     "used_generations": 0, "max_generations": 1})
    forwarded = []

    def upstream(method, path, **kwargs):
        forwarded.append((method, path, kwargs))
        response = requests.Response()
        response.status_code = 202 if method == "POST" else 200
        response.headers["Content-Type"] = "application/json"
        runs = [{"id": rid, "case_id": "coffee-clean", "status": "succeeded", "artifact_path": "PRIVATE_PATH",
                 "provider": {"id": "PRIVATE_PROVIDER_ID"}, "usage": {"private": True}} for rid in ["demo", "private-local", "student-run"]]
        body = {"id": "student-run"} if method == "POST" else {"runs": runs, "busy": False, "token": "csrf",
                                                                               "provider": {"configured": True, "id": "PRIVATE_PROVIDER_ID"}}
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(share, "local_request", upstream)
    server = share.LocalServer(("127.0.0.1", 0), share.ShareHandler)
    server.share = share.ShareState(path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def join():
        r = requests.post(url+"/api/join", json={"invite": "test-invitation"}, headers={"Origin": "https://class.example"}, timeout=3)
        assert r.status_code == 200
        cookie = SimpleCookie(r.headers["Set-Cookie"])
        return {"Cookie": f"{share.COOKIE}={cookie[share.COOKIE].value}", "Origin": "https://class.example", "X-Fusion-Token": "csrf"}, r

    yield url, server.share, forwarded, join
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_invite_gate_denies_api_without_cookie_and_rejects_foreign_origin(gateway):
    url, _, forwarded, _ = gateway
    assert requests.get(url+"/api/state", timeout=3).status_code == 401
    assert requests.post(url+"/api/join", json={"invite": "test-invitation"}, headers={"Origin": "https://evil.example"}, timeout=3).status_code == 403
    assert requests.post(url+"/api/join", json={"invite": "wrong"}, headers={"Origin": "https://class.example"}, timeout=3).status_code == 401
    assert forwarded == []


def test_session_cookie_and_state_do_not_expose_local_details(gateway):
    url, _, _, join = gateway
    headers, joined = join()
    cookie = joined.headers["Set-Cookie"]
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    r = requests.get(url+"/api/state", headers=headers, timeout=3)
    assert r.status_code == 200
    assert [x["id"] for x in r.json()["runs"]] == ["demo"]
    assert "PRIVATE_" not in r.text and "artifact_path" not in r.text and "usage" not in r.text


def test_guest_tasks_are_private_and_total_budget_cannot_be_exceeded(gateway):
    url, state, forwarded, join = gateway
    first, _ = join()
    second, _ = join()
    r = requests.post(url+"/api/run", headers=first, json={"case_id": "coffee-clean", "dry_run": False}, timeout=3)
    assert r.status_code == 202
    assert state.load()["used_generations"] == 1
    assert requests.post(url+"/api/run", headers=second, json={"dry_run": False}, timeout=3).status_code == 429
    assert [r["id"] for r in requests.get(url+"/api/runs", headers=first, timeout=3).json()["runs"]] == ["demo", "student-run"]
    assert [r["id"] for r in requests.get(url+"/api/runs", headers=second, timeout=3).json()["runs"]] == ["demo"]
    assert requests.get(url+"/runs/student-run/result.png", headers=second, timeout=3).status_code == 403
    assert sum(method == "POST" for method, _, _ in forwarded) == 1


def test_dry_run_costs_no_shared_quota_and_never_forwards_target_overrides(gateway):
    url, state, forwarded, join = gateway
    headers, _ = join()
    r = requests.post(url+"/api/run", headers=headers, json={"dry_run": True, "provider_id": "evil", "base_url": "https://evil.example"}, timeout=3)
    assert r.status_code == 202
    assert state.load()["used_generations"] == 0
    body = next(k["json"] for method, _, k in forwarded if method == "POST")
    assert body == {"dry_run": True}


def test_private_run_metadata_and_arbitrary_files_cannot_be_downloaded(gateway):
    url, _, _, join = gateway
    headers, _ = join()
    for path in ["/runs/demo/run.json", "/runs/demo/prompt.txt", "/runs/private-local/result.png", "/runs/demo/../../auth.json"]:
        assert requests.get(url+path, headers=headers, timeout=3).status_code in {403, 404}
    assert requests.get(url+"/state.json", headers=headers, timeout=3).status_code == 404


def test_expiry_and_stop_apply_even_to_existing_authenticated_sessions(gateway):
    url, state, _, join = gateway
    headers, _ = join()
    state.change(lambda d: d.__setitem__("expires_at", time.time()-1))
    assert requests.get(url+"/api/state", headers=headers, timeout=3).status_code == 410
    state.change(lambda d: d.__setitem__("expires_at", time.time()+3600))
    (state.path.parent/"STOP").touch()
    assert requests.get(url+"/api/state", headers=headers, timeout=3).status_code == 410


def test_backend_rejection_refunds_reservation(gateway, monkeypatch):
    url, state, _, join = gateway
    headers, _ = join()
    def busy(*args, **kwargs):
        r = requests.Response(); r.status_code = 409; r._content = b'{"error":"busy"}'; return r
    monkeypatch.setattr(share, "local_request", busy)
    assert requests.post(url+"/api/run", headers=headers, json={"dry_run": False}, timeout=3).status_code == 409
    assert state.load()["used_generations"] == 0


def test_ambiguous_timeout_keeps_reserved_slot_to_prevent_double_spend(gateway, monkeypatch):
    url, state, _, join = gateway
    headers, _ = join()
    def timeout(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr(share, "local_request", timeout)
    assert requests.post(url+"/api/run", headers=headers, json={"dry_run": False}, timeout=3).status_code == 503
    assert state.load()["used_generations"] == 1
