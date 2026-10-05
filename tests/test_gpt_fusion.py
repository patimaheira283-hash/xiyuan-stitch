import base64
import io
import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from gpt_fusion_mvp import core, responses_client as api
from gpt_fusion_mvp.server import safe_child


@pytest.fixture
def images(tmp_path):
    paths = []
    for name, color in (("left", "red"), ("right", "blue"), ("reference", "green")):
        path = tmp_path / (name + ".png")
        Image.new("RGB", (96, 64), color).save(path)
        paths.append(path)
    return paths


@pytest.fixture
def provider():
    return core.Provider("test", "Test Provider", "https://provider.example/v1", "sk-unit-secret-123", "gpt-6-astra")


def test_cc_switch_selects_current_provider_without_writing_db(tmp_path):
    path = tmp_path / "cc-switch.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE providers(id TEXT,app_type TEXT,name TEXT,settings_config TEXT,is_current INTEGER)")
        config = json.dumps({"auth": {"OPENAI_API_KEY": "sk-test-only"}, "config": 'model="gpt-6-astra"\nmodel_provider="custom"\n[model_providers.custom]\nbase_url="https://provider.example/v1"'})
        conn.execute("INSERT INTO providers VALUES('active','codex','Active',?,1)", (config,))
        conn.execute("INSERT INTO providers VALUES('other','codex','Other',?,0)", (config,))
    before = path.read_bytes()
    p = core.load_provider(db_path=path)
    assert p.id == "active"
    assert "sk-test-only" not in json.dumps(p.public())
    assert "sk-test-only" not in repr(p)
    assert path.read_bytes() == before


def test_direct_request_contains_only_two_originals_and_explicit_image_model(images):
    left, right, reference = images
    request = api.build_request(left, right, "stitch", "gpt-6-astra", core.MODEL, "high", "1536x1024")
    image_inputs = [i for i in request["input"][0]["content"] if i["type"] == "input_image"]
    assert len(image_inputs) == 2
    assert [base64.b64decode(i["image_url"].split(",", 1)[1]) for i in image_inputs] == [left.read_bytes(), right.read_bytes()]
    assert base64.b64encode(reference.read_bytes()).decode() not in json.dumps(request)
    assert request["tools"][0]["model"] == core.MODEL
    assert "mask" not in json.dumps(request)


def test_stream_parser_handles_keepalive_and_multiline_frames():
    raw = [b":keep-alive", b"", b"event: response.completed", b'data: {"type":', b'data: "response.completed"}', b"", b"data: [DONE]", b""]
    assert list(api.events(raw)) == [{"type": "response.completed"}]


class FakeResponse:
    ok = True
    status_code = 200
    headers = {"Content-Type": "text/event-stream", "x-request-id": "test-request"}

    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self):
        for event in self.stream:
            yield ("data: " + json.dumps(event)).encode()
            yield b""


def test_stream_deduplicates_image_in_done_and_completed(monkeypatch, tmp_path, images, provider):
    b64 = base64.b64encode(images[0].read_bytes()).decode()
    item = {"type": "image_generation_call", "id": "ig_test", "status": "completed", "result": b64}
    stream = [{"type": "response.output_item.done", "item": item}, {"type": "response.completed", "response": {"output": [item], "model": "gpt-6-astra", "usage": {"total_tokens": 42}}}]
    monkeypatch.setattr(api.requests, "post", lambda *a, **k: FakeResponse(stream))
    result = api.generate(provider, {}, tmp_path / "output.png")
    assert len(result["artifacts"]) == 1
    assert Path(result["artifacts"][0]).read_bytes() == images[0].read_bytes()
    assert result["usage"]["total_tokens"] == 42


def test_successful_image_survives_later_stream_error(monkeypatch, tmp_path, images, provider):
    item = {"type": "image_generation_call", "result": base64.b64encode(images[0].read_bytes()).decode()}
    stream = [{"type": "response.output_item.done", "item": item}, {"type": "error", "message": "sk-unit-secret-123 transport error"}]
    monkeypatch.setattr(api.requests, "post", lambda *a, **k: FakeResponse(stream))
    result = api.generate(provider, {}, tmp_path / "output.png")
    assert len(result["artifacts"]) == 1
    assert "transport error" in result["partial_error"]
    assert provider.api_key not in result["partial_error"]


def test_text_response_is_not_misreported_as_image_success(monkeypatch, tmp_path, provider):
    stream = [{"type": "response.completed", "response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Cannot make image"}]}]}}]
    monkeypatch.setattr(api.requests, "post", lambda *a, **k: FakeResponse(stream))
    with pytest.raises(RuntimeError, match="没有返回"):
        api.generate(provider, {}, tmp_path / "output.png")
    assert not (tmp_path / "output.png").exists()


@pytest.mark.parametrize("path", ["../auth.json", "../../.cc-switch/cc-switch.db"])
def test_server_cannot_serve_files_outside_asset_folder(tmp_path, path):
    with pytest.raises(ValueError):
        safe_child(tmp_path / "assets", path)


def test_run_dry_run_never_posts_or_stores_credentials(monkeypatch, tmp_path, images, provider):
    monkeypatch.setattr(core, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(core, "artifact_root", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(core, "load_provider", lambda *a: provider)
    monkeypatch.setattr(api.requests, "post", lambda *a, **k: pytest.fail("dry-run made a network call"))
    r = core.run_fusion(images[0], images[1], truth=images[2], dry_run=True)
    assert r["status"] == "dry_run"
    for path in (tmp_path / "runs").rglob("*.json"):
        assert provider.api_key not in path.read_text(encoding="utf-8")
    preview = json.loads((tmp_path / "runs" / r["id"] / "request-preview.json").read_text(encoding="utf-8"))
    assert all("base64," not in i.get("image_url", "") for i in preview["input"][0]["content"])
