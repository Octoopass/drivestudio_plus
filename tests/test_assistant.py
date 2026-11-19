"""In-app assistant: per-provider key storage (never returned), OpenAI-compatible chat against a local fake
server, the Anthropic path (module stubbed), and parsing of the "fflab-apply" proposal."""
import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import fisheye_fusion.assistant as A


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "assistant.json"
    monkeypatch.setattr(A, "STORE", path)
    monkeypatch.delenv("FFLAB_ASSISTANT_KEY", raising=False)
    monkeypatch.delenv("FFLAB_ASSISTANT_URL", raising=False)
    return path


@pytest.fixture
def fake_api():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen["auth"] = self.headers.get("Authorization")
            self._send({"data": [{"id": "luna-6"}, {"id": "other"}]})

        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = 'Dùng streetgs_ut.\n```fflab-apply\n{"method": "configs/streetgs_ut.yaml", "steps": 3000, "evil": 1}\n```'
            self._send({"model": "luna-6", "choices": [{"message": {"content": reply}}]})

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", seen
    srv.shutdown()


def test_keys_per_provider_and_never_returned(store):
    a = A.Assistant(store.parent, lambda form: {})
    cfg = a.set_config(provider="custom", base_url="https://example.test/v1/", api_key="sk-secret-custom-1234", model="luna-6")
    assert cfg["has_key"] and cfg["key_hint"] == "…1234" and cfg["base_url"] == "https://example.test/v1"
    assert "sk-secret" not in json.dumps(cfg)
    cfg = a.set_config(provider="anthropic", api_key="sk-ant-secret-9999")
    assert cfg["provider"] == "anthropic" and cfg["model"] == "claude-opus-5-5" and cfg["key_hint"] == "…9999"
    cfg = a.set_config(provider="custom")                       # switching back keeps the custom key + model
    assert cfg["key_hint"] == "…1234" and cfg["model"] == "luna-6"
    assert stat.S_IMODE(os.stat(store).st_mode) == 0o600
    with pytest.raises(ValueError):
        a.set_config(base_url="not a url")
    with pytest.raises(ValueError):
        a.set_config(provider="nope")
    assert not a.set_config(clear_key=True)["has_key"]


def test_openai_compatible_chat_sends_app_facts(store, fake_api):
    url, seen = fake_api
    facts = {"prepared_scenes": [{"data_root": "data/kitti360/processed_x", "cameras": {"2": {"fisheye": True}}}]}
    a = A.Assistant(store.parent, lambda form: {**facts, "form": form})
    a.set_config(provider="custom", base_url=url, api_key="sk-test-abcdefgh")
    assert a.models()["models"] == ["luna-6", "other"] and seen["auth"] == "Bearer sk-test-abcdefgh"
    a.set_config(model="luna-6")
    r = a.chat([{"role": "user", "content": "Nên train gì?"}], form={"method": "configs/streetgs.yaml"}, page="step-train")
    assert r["model"] == "luna-6" and r["apply"] == {"method": "configs/streetgs_ut.yaml", "steps": 3000}
    msgs = seen["body"]["messages"]
    assert [m["role"] for m in msgs].count("system") == 1 and msgs[0]["role"] == "system"
    assert "fflab-apply" in msgs[0]["content"] and "processed_x" in msgs[0]["content"]
    assert "configs/streetgs.yaml" in msgs[0]["content"] and "step-train" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "Nên train gì?"}


def test_anthropic_path_uses_sdk_module(store, monkeypatch):
    import fisheye_fusion.assistant_anthropic as AA
    calls = {}

    def fake_chat(key, model, system, messages):
        calls.update(key=key, model=model, system=system, messages=messages)
        return {"reply": "ok", "model": model}
    monkeypatch.setattr(AA, "chat", fake_chat)
    a = A.Assistant(store.parent, lambda form: {"gpus": []})
    a.set_config(provider="anthropic", api_key="sk-ant-test-0000")
    r = a.chat([{"role": "user", "content": "hi"}])
    assert r == {"reply": "ok", "model": "claude-opus-5-5", "apply": None}
    assert calls["model"] == "claude-opus-5-5" and "DỮ KIỆN APP" in calls["system"] and calls["messages"][0]["role"] == "user"


def test_errors_without_setup(store):
    a = A.Assistant(store.parent, lambda form: {})
    with pytest.raises(ValueError):
        a.chat([{"role": "user", "content": "x"}])
    a.set_config(provider="custom", base_url="http://127.0.0.1:9/v1", model="m")
    with pytest.raises(ValueError, match="key"):
        a.chat([{"role": "user", "content": "x"}])
    assert A.parse_apply("no block") is None and A.parse_apply("```fflab-apply\n{bad json}\n```") is None
