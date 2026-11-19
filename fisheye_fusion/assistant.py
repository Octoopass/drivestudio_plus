"""In-app assistant answering with the app's real state as context, and able to propose a training form
("fflab-apply" block) the UI can fill in.

Providers: an official API (OpenAI, Anthropic, Google Gemini, OpenRouter) or a custom OpenAI-compatible endpoint.
Anthropic goes through the official SDK (assistant_anthropic.py); the others share the OpenAI chat format.
API keys are typed by the user in the UI, one per provider, and stored server-side in
~/.config/fflab/assistant.json (mode 600); FFLAB_ASSISTANT_KEY overrides the current provider's key.
Keys are never returned to the browser (only a masked hint).

    GET  /api/assistant/config    {provider, providers, base_url, model, has_key, key_hint}
    POST /api/assistant/config    {provider?, base_url?, model?, api_key?, clear_key?}
    GET  /api/assistant/models    model ids from <base_url>/models
    POST /api/assistant/chat      {messages: [{role, content}], form?: {...}, page?: str} -> {reply, apply?}
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

STORE = Path(os.environ.get("FFLAB_ASSISTANT_STORE", Path.home() / ".config" / "fflab" / "assistant.json"))
PROVIDERS = {   # id -> label, fixed base URL (None = the user enters it), suggested model
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "model": ""},
    "anthropic": {"label": "Anthropic (Claude)", "base_url": None, "model": "claude-opus-5-5"},
    "gemini": {"label": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": ""},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": ""},
    "custom": {"label": "Custom endpoint (OpenAI-compatible)", "base_url": None, "model": ""},
}
APPLY_KEYS = {"dataset", "data_root", "method", "render_mode", "fastgs", "steps", "gpu", "target", "name"}
SYSTEM = """Bạn là trợ lý của Fisheye Fusion Lab Studio, web app dựng cảnh 3D Gaussian Splatting từ dữ liệu lái xe
(DriveStudio+: Street Gaussians, OmniRe, PVG, DeformableGS; renderer 3DGS pinhole / 3DGUT / 3DGEER; FastGS; Instant NuRec).
Trả lời bằng tiếng Việt, ngắn gọn, cụ thể, chỉ dựa vào DỮ KIỆN APP bên dưới và kiến thức chung về 3DGS.
Không bịa tên config, đường dẫn, số liệu: nếu dữ kiện không có thì nói là không biết.

Quy tắc chọn cấu hình train (dùng khi được hỏi nên chạy gì):
- Dữ liệu có camera fisheye (KITTI-360 image_02/03, PhysicalAI cross_*) bắt buộc renderer `ut` hoặc `geer`; config không đặt render_mode sẽ chạy pinhole và render fisheye sai.
- Có 3D object tracks thì ưu tiên streetgs*; omnire* chỉ có lợi khi config dataset bật SMPL (load_smpl: True); pvg/deformablegs bỏ qua box.
- Config ghi "Untested" là thử nghiệm: khuyên chạy 3000 bước thử trước.
- Lần đầu với một bộ dữ liệu: chạy ngắn (3000 bước) trước, rồi 30000.
- Chọn GPU đang rảnh; có thể chạy 2 cấu hình song song trên 2 GPU để so sánh.

Khi đề xuất một cấu hình train cụ thể, thêm ở CUỐI câu trả lời đúng một khối:
```fflab-apply
{"dataset": "...", "data_root": "...", "method": "configs/....yaml", "render_mode": "", "fastgs": "", "steps": 3000, "gpu": "0", "target": "local"}
```
Chỉ dùng giá trị có trong DỮ KIỆN APP. render_mode: "" | "ut" | "geer" | "default" ("" = theo config).
fastgs: "" | "true" | "false". target: "local" (GPU máy chủ) hoặc "colab". Bỏ khối này nếu không đề xuất cấu hình."""


class Assistant:
    def __init__(self, workspace: Path, gather):
        self.workspace = Path(workspace)
        self.gather = gather            # callable(form: dict) -> dict of app facts

    # ---- config / key storage
    def _load(self) -> dict:
        try:
            data = json.loads(STORE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("provider", "custom")
        data.setdefault("keys", {})
        data.setdefault("models", {})
        if data.get("api_key"):                     # older single-key layout
            data["keys"].setdefault("custom", data.pop("api_key"))
        if data.get("model") and not data["models"].get(data["provider"]):
            data["models"][data["provider"]] = data.pop("model")
        if os.environ.get("FFLAB_ASSISTANT_URL") and not data.get("base_url"):
            data["base_url"] = os.environ["FFLAB_ASSISTANT_URL"]
        return data

    def _save(self, data: dict):
        STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        keep = {k: v for k, v in data.items() if k in ("provider", "base_url", "keys", "models") and v}
        tmp.write_text(json.dumps(keep, indent=1), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(STORE)

    def _current(self) -> dict:
        """Effective settings for the selected provider: provider, base_url, model, api_key."""
        d = self._load()
        prov = d["provider"] if d["provider"] in PROVIDERS else "custom"
        base = PROVIDERS[prov]["base_url"] if prov != "custom" else d.get("base_url", "")
        key = os.environ.get("FFLAB_ASSISTANT_KEY") or d["keys"].get(prov, "")
        model = d["models"].get(prov) or PROVIDERS[prov]["model"]
        return {"provider": prov, "base_url": base or "", "model": model, "api_key": key}

    def config(self) -> dict:
        c = self._current()
        key = c["api_key"]
        return {"provider": c["provider"], "base_url": c["base_url"], "model": c["model"], "has_key": bool(key),
                "key_hint": f"…{key[-4:]}" if len(key) >= 8 else ("đã có" if key else ""),
                "providers": [{"id": k, "label": v["label"], "base_url": v["base_url"], "needs_url": k == "custom"}
                              for k, v in PROVIDERS.items()]}

    def set_config(self, provider=None, base_url=None, model=None, api_key=None, clear_key=False) -> dict:
        d = self._load()
        if provider is not None:
            if provider not in PROVIDERS:
                raise ValueError("Nhà cung cấp không hợp lệ")
            d["provider"] = provider
        prov = d["provider"]
        if base_url is not None and prov == "custom":
            base_url = base_url.strip().rstrip("/")
            if base_url and not re.match(r"^https?://[^\s/]+(/[^\s]*)?$", base_url):
                raise ValueError("URL không hợp lệ (vd. https://host/v1)")
            d["base_url"] = base_url
        if model is not None:
            d["models"][prov] = model.strip()[:120]
        if clear_key:
            d["keys"].pop(prov, None)
        elif api_key:
            d["keys"][prov] = api_key.strip()
        self._save(d)
        return self.config()

    # ---- HTTP (OpenAI-compatible providers)
    def _request(self, path: str, body: dict | None = None, timeout: float = 120) -> dict:
        c = self._current()
        if not c["base_url"]:
            raise ValueError("Chưa đặt URL của API trợ lý (bấm biểu tượng cài đặt trong khung Trợ lý)")
        if not c["api_key"]:
            raise ValueError("Chưa nhập API key cho trợ lý")
        req = urllib.request.Request(c["base_url"] + path, data=None if body is None else json.dumps(body).encode(),
                                     method="GET" if body is None else "POST",
                                     headers={"Authorization": f"Bearer {c['api_key']}", "Content-Type": "application/json",
                                              "User-Agent": "fflab-assistant"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise ValueError(f"API trợ lý trả lỗi {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ValueError(f"Không gọi được API trợ lý: {e}") from e

    def models(self) -> dict:
        c = self._current()
        if c["provider"] == "anthropic":
            if not c["api_key"]:
                raise ValueError("Chưa nhập API key Anthropic")
            from . import assistant_anthropic
            return {"models": assistant_anthropic.list_models(c["api_key"])}
        data = self._request("/models", timeout=30)
        ids = sorted({m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")})
        return {"models": ids}

    # ---- chat
    def chat(self, messages: list[dict], form: dict | None = None, page: str = "") -> dict:
        c = self._current()
        if not c["model"]:
            raise ValueError("Chưa chọn model cho trợ lý")
        clean = [{"role": m["role"], "content": str(m["content"])[:6000]} for m in messages[-12:]
                 if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]
        if not clean or clean[-1]["role"] != "user":
            raise ValueError("Thiếu câu hỏi")
        facts = self.gather(form or {})
        context = "DỮ KIỆN APP (JSON, cập nhật lúc hỏi):\n" + json.dumps(facts, ensure_ascii=False, default=str)[:24000]
        if page:
            context += f"\nNgười dùng đang ở trang: {page}"
        if c["provider"] == "anthropic":
            if not c["api_key"]:
                raise ValueError("Chưa nhập API key Anthropic")
            from . import assistant_anthropic
            r = assistant_anthropic.chat(c["api_key"], c["model"], SYSTEM + "\n\n" + context, clean)
            return {**r, "apply": parse_apply(r["reply"])}
        payload = {"model": c["model"], "temperature": 0.3,
                   # one system message: some OpenAI-compatible gateways keep only the first one
                   "messages": [{"role": "system", "content": SYSTEM + "\n\n" + context}, *clean]}
        data = self._request("/chat/completions", payload)
        try:
            reply = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"API trợ lý trả kết quả lạ: {str(data)[:300]}") from e
        return {"reply": reply, "apply": parse_apply(reply), "model": data.get("model", c["model"])}


def parse_apply(text: str) -> dict | None:
    m = re.search(r"```fflab-apply\s*(\{.*?\})\s*```", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except ValueError:
        return None
    return {k: v for k, v in obj.items() if k in APPLY_KEYS} if isinstance(obj, dict) else None


def config_headers(path: Path, lines: int = 6) -> str:
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[:lines]:
            if line.startswith("#"):
                out.append(line.lstrip("# ").strip())
    except OSError:
        pass
    return " ".join(out)[:400]
