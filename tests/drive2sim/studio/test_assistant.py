"""Test trợ lý của Studio (``drive2sim/studio/api/assistant.py`` và route).

Không gọi nhà cung cấp thật: endpoint chuẩn OpenAI thay bằng ``urlopen`` giả, Anthropic thay bằng
một gói ``anthropic`` giả có đủ tên lớp lỗi mà mã dùng.
"""

import io
import json
import os
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api import assistant
from drive2sim.studio.api.app import create_app


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cfg" / "assistant.json"
    monkeypatch.setenv("D2S_ASSISTANT_STORE", str(path))
    monkeypatch.delenv("D2S_ASSISTANT_KEY", raising=False)
    return path


def test_config_keeps_key_secret(store: Path) -> None:
    assert assistant.config()["provider"] == "anthropic" and not assistant.config()["has_key"]
    with pytest.raises(ValueError):
        assistant.set_config(provider="nope")
    assistant.set_config(provider="custom", base_url="https://llm.example/v1/", model="m1")
    shown = assistant.set_config(api_key="sk-test-123456789")
    assert shown["key_hint"] == "…6789" and "sk-test" not in json.dumps(shown)
    assert shown["base_url"] == "https://llm.example/v1" and shown["model"] == "m1"
    if os.name == "posix":
        assert store.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="URL"):
        assistant.set_config(base_url="ftp://x")
    assert not assistant.set_config(clear_key=True)["has_key"]


def test_parse_apply() -> None:
    text = 'Chạy thử:\n```d2s-apply\n{"scene_id": "a", "approach_id": "omnire", "bad": 1}\n```'
    assert assistant.parse_apply(text) == {"scene_id": "a", "approach_id": "omnire"}
    assert assistant.parse_apply("```d2s-apply\n{hỏng}\n```") is None
    assert assistant.parse_apply("không có") is None


def test_chat_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    assistant.set_config(provider="custom", base_url="https://llm.example/v1", model="m1")
    assistant.set_config(api_key="sk-test-123456789")
    sent = {}

    def urlopen(request, timeout=0):  # noqa: ANN001, ANN202
        sent["url"] = request.full_url
        sent["auth"] = request.headers["Authorization"]
        sent["body"] = json.loads(request.data)
        reply = {"model": "m1", "choices": [{"message": {"content": "Dùng omnire."}}]}
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(assistant.urllib.request, "urlopen", urlopen)
    result = assistant.chat([{"role": "user", "content": "chạy gì?"}], {"scenes": []}, "new")
    assert result == {"reply": "Dùng omnire.", "apply": None, "model": "m1"}
    assert sent["url"] == "https://llm.example/v1/chat/completions"
    assert sent["auth"] == "Bearer sk-test-123456789"
    system = sent["body"]["messages"][0]
    assert system["role"] == "system" and "DỮ KIỆN STUDIO" in system["content"]
    with pytest.raises(ValueError, match="thiếu câu hỏi"):
        assistant.chat([{"role": "assistant", "content": "x"}], {}, "")


def fake_anthropic(stop_reason: str = "end_turn") -> types.ModuleType:
    module = types.ModuleType("anthropic")
    calls = []

    class APIStatusError(Exception):
        status_code, message = 500, "lỗi"

    module.AuthenticationError = type("AuthenticationError", (APIStatusError,), {})
    module.RateLimitError = type("RateLimitError", (APIStatusError,), {})
    module.APIStatusError = APIStatusError
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})

    def create(**kwargs: object) -> object:
        calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text="Chọn streetgs.")
        return types.SimpleNamespace(
            stop_reason=stop_reason, content=[block], model=kwargs["model"], stop_details=None
        )

    messages = types.SimpleNamespace(create=create)

    class Anthropic:
        def __init__(self, **kwargs: object) -> None:
            self.messages = messages
            self.beta = types.SimpleNamespace(messages=messages)
            self.models = types.SimpleNamespace(
                list=lambda: [types.SimpleNamespace(id="claude-opus-5-5")]
            )

    module.Anthropic = Anthropic
    module.calls = calls
    return module


def test_chat_anthropic_uses_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    module = fake_anthropic()
    monkeypatch.setitem(sys.modules, "anthropic", module)
    assistant.set_config(provider="anthropic", api_key="sk-ant-test-1234")
    result = assistant.chat([{"role": "user", "content": "chạy gì?"}], {}, "")
    assert result["reply"] == "Chọn streetgs." and result["model"] == "claude-opus-5-5"
    call = module.calls[-1]
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "low"} and "DỮ KIỆN" in call["system"]
    assert assistant.models() == {"models": ["claude-opus-5-5"]}
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic("refusal"))
    with pytest.raises(ValueError, match="refusal"):
        assistant.chat([{"role": "user", "content": "x"}], {}, "")


def test_anthropic_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assistant.set_config(provider="anthropic", api_key="sk-ant-test-1234")
    with pytest.raises(ValueError, match="pip install anthropic"):
        assistant.models()


def test_routes_and_facts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(Workdir(tmp_path / "work"), registry=StageRegistry(), frontend_dir=None)
    seen = {}

    def chat(messages: list, facts: dict, page: str) -> dict:
        seen.update(facts=facts, page=page)
        return {"reply": "ok", "apply": None, "model": "m"}

    monkeypatch.setattr(assistant, "chat", chat)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/assistant/config").json()["has_key"] is False
        saved = client.post(
            "/api/assistant/config", json={"provider": "custom", "api_key": "abcdefgh1"}
        )
        assert saved.json()["key_hint"] == "…fgh1" and "abcdefgh1" not in saved.text
        assert client.post("/api/assistant/config", json={"provider": "x"}).status_code == 400
        reply = client.post(
            "/api/assistant/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "page": "new"},
        )
        assert reply.status_code == 200 and reply.json()["reply"] == "ok"
    assert seen["page"] == "new" and seen["facts"]["scenes"] == []
    assert {a["approach_id"] for a in seen["facts"]["approaches"]} >= {"omnire"}
