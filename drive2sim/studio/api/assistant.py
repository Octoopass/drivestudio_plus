"""Trợ lý hỏi đáp trong Studio, trả lời dựa trên trạng thái thật của thư mục làm việc.

Người dùng mới hay hỏi "cảnh này nên chạy approach nào, preset nào, trên máy hay Colab". Trợ lý
gửi câu hỏi kèm dữ kiện của Studio lúc hỏi (cảnh, split, approach và preset, run gần đây, môi
trường, GPU; không có khoá, không có đường dẫn tuyệt đối) tới một mô hình ngôn ngữ, và có thể đề
xuất một cấu hình chạy dạng khối ``d2s-apply`` để giao diện mở trang tạo run với lựa chọn đó.

Nhà cung cấp: API chính thức (Anthropic, OpenAI, Google Gemini, OpenRouter) hoặc một endpoint tự
đặt theo chuẩn OpenAI (``/chat/completions``, ``/models``). Anthropic đi qua SDK chính thức
``anthropic`` (gói tuỳ chọn, chỉ cần khi chọn nhà cung cấp này); các nhà cung cấp khác dùng
``urllib``. API key người dùng nhập trên Studio, mỗi nhà cung cấp một key, lưu ở
``~/.config/drive2sim/assistant.json`` (quyền 600; ``D2S_ASSISTANT_STORE`` đổi đường dẫn), không
bao giờ trả về trình duyệt (chỉ 4 ký tự cuối). ``D2S_ASSISTANT_KEY`` thay key của nhà cung cấp
đang chọn.

File này chứa:
- ``PROVIDERS``, ``SYSTEM``: nhà cung cấp và lời dặn cho mô hình.
- ``config``, ``set_config``: đọc, ghi cài đặt.
- ``models``, ``chat``: danh sách model của nhà cung cấp, hỏi đáp.
- ``gather_facts``, ``parse_apply``: dữ kiện gửi kèm câu hỏi, đọc khối đề xuất.
"""

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, TypeVar

from drive2sim.engine import scenes
from drive2sim.engine.approaches import list_approach_ids, load_approach
from drive2sim.engine.environment import environment_check
from drive2sim.engine.machine import _gpus
from drive2sim.engine.run import list_runs
from drive2sim.engine.workdir import Workdir

PROVIDERS = {
    "anthropic": {"label": "Anthropic (Claude)", "base_url": None, "model": "claude-opus-5-5"},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "model": ""},
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "",
    },
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": ""},
    "custom": {"label": "Endpoint tự đặt (chuẩn OpenAI)", "base_url": None, "model": ""},
}
# Model Claude nhận ``fallbacks: "default"`` (chạy lại trên model khác khi bị từ chối) và effort.
CLAUDE_FALLBACK = {
    "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5",
    "claude-sonnet-5-5",
}  # fmt: skip
CLAUDE_EFFORT = CLAUDE_FALLBACK | {"claude-haiku-5-5"}
APPLY_KEYS = {"scene_id", "split_id", "approach_id", "preset", "executor", "quick"}
URL_PATTERN = r"^https?://[^\s/]+(/\S*)?$"
MAX_FACTS = 24000
T = TypeVar("T")
SYSTEM = """Bạn là trợ lý của drive2sim Studio, công cụ dựng lại cảnh lái xe thành cảnh 3D Gaussian
Splatting để mô phỏng. Luồng dùng: nạp cảnh từ dataset (nuScenes, KITTI-360, PandaSet...) → tạo
split (chọn camera, khung train/test) → chọn approach (OmniRe, Street Gaussians...) và preset →
chạy trên máy này, Kaggle hoặc Colab → xem metric, video, cảnh 3D, so sánh run.

Trả lời bằng tiếng Việt, ngắn gọn, cụ thể, chỉ dựa vào DỮ KIỆN STUDIO bên dưới và kiến thức chung
về 3DGS. Không bịa tên cảnh, split, approach, preset, đường dẫn hay số liệu: dữ kiện không có thì
nói là không biết.

Khi chọn cấu hình chạy:
- Split có camera mắt cá (fisheye) cần preset vẽ được camera đó (xem camera_support của approach:
  preset nào null là vẽ được; renderer 3DGUT hoặc 3DGEER), không thì run bị khoá.
- OmniRe dựng người bằng SMPL: cần file SMPL_NEUTRAL.pkl (environment.checks smpl_model) hoặc chọn
  nguồn SMPL khác; Street Gaussians không cần.
- Lần đầu với một cảnh: chạy "thử nhanh" (quick) trước rồi mới chạy đầy đủ.
- Máy này thiếu môi trường (environment.level missing) hoặc không có GPU thì đề xuất Kaggle hoặc
  Colab (executor kaggle, colab).

Khi đề xuất một cấu hình cụ thể, thêm ở CUỐI câu trả lời đúng một khối:
```d2s-apply
{"scene_id": "...", "split_id": "...", "approach_id": "...", "preset": "",
 "executor": "local", "quick": true}
```
Chỉ dùng giá trị có trong dữ kiện. preset "" là mặc định của approach. executor: local, kaggle,
colab. Không đề xuất cấu hình thì bỏ khối này."""


def _store() -> Path:
    default = Path.home() / ".config" / "drive2sim" / "assistant.json"
    return Path(os.environ.get("D2S_ASSISTANT_STORE", default))


def _load() -> dict:
    try:
        data = json.loads(_store().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data = data if isinstance(data, dict) else {}
    data.setdefault("provider", "anthropic")
    data.setdefault("keys", {})
    data.setdefault("models", {})
    return data


def _save(data: dict) -> None:
    path = _store()
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in data.items() if k in ("provider", "base_url", "keys", "models") and v}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(keep, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def _current() -> dict:
    data = _load()
    provider = data["provider"] if data["provider"] in PROVIDERS else "custom"
    base = PROVIDERS[provider]["base_url"] or (
        data.get("base_url", "") if provider == "custom" else ""
    )
    key = os.environ.get("D2S_ASSISTANT_KEY") or data["keys"].get(provider, "")
    model = data["models"].get(provider) or PROVIDERS[provider]["model"]
    return {"provider": provider, "base_url": base or "", "model": model, "api_key": key}


def config() -> dict:
    """Cài đặt hiện tại, không có key.

    Returns:
        ``{"provider", "base_url", "model", "has_key", "key_hint", "providers": [{"id",
        "label", "base_url", "needs_url"}]}``.
    """
    current = _current()
    key = current["api_key"]
    return {
        "provider": current["provider"],
        "base_url": current["base_url"],
        "model": current["model"],
        "has_key": bool(key),
        "key_hint": f"…{key[-4:]}" if len(key) >= 8 else ("có" if key else ""),
        "providers": [
            {"id": k, "label": v["label"], "base_url": v["base_url"], "needs_url": k == "custom"}
            for k, v in PROVIDERS.items()
        ],
    }


def set_config(
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    clear_key: bool = False,
) -> dict:
    """Đổi nhà cung cấp, URL (endpoint tự đặt), model, key của nhà cung cấp đang chọn.

    Returns:
        ``config()`` sau khi đổi.

    Raises:
        ValueError: Nhà cung cấp lạ hoặc URL sai dạng.
    """
    data = _load()
    if provider is not None:
        if provider not in PROVIDERS:
            raise ValueError(f"nhà cung cấp không hợp lệ: {provider}")
        data["provider"] = provider
    current = data["provider"]
    if base_url is not None and current == "custom":
        base_url = base_url.strip().rstrip("/")
        if base_url and not re.match(URL_PATTERN, base_url):
            raise ValueError("URL không hợp lệ (ví dụ https://host/v1)")
        data["base_url"] = base_url
    if model is not None:
        data["models"][current] = model.strip()[:120]
    if clear_key:
        data["keys"].pop(current, None)
    elif api_key:
        data["keys"][current] = api_key.strip()
    _save(data)
    return config()


def _request(current: dict, path: str, body: Optional[dict] = None, timeout: float = 120) -> dict:
    if not current["base_url"]:
        raise ValueError("chưa đặt URL của API trợ lý (mở Cài đặt trong ngăn Trợ lý)")
    if not current["api_key"]:
        raise ValueError("chưa nhập API key cho trợ lý")
    request = urllib.request.Request(
        current["base_url"] + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        method="GET" if body is None else "POST",
        headers={
            "Authorization": f"Bearer {current['api_key']}",
            "Content-Type": "application/json",
            "User-Agent": "drive2sim-studio",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:400]
        raise ValueError(f"API trợ lý trả lỗi {error.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        raise ValueError(f"không gọi được API trợ lý: {error}") from None


def _anthropic_client(api_key: str) -> tuple[ModuleType, object]:
    if not api_key:
        raise ValueError("chưa nhập API key Anthropic")
    try:
        import anthropic
    except ImportError:
        raise ValueError(
            "nhà cung cấp Anthropic cần gói anthropic: cài vào Python đang chạy Studio "
            "(pip install anthropic) rồi mở lại Studio"
        ) from None
    return anthropic, anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=2)


def _anthropic_errors(anthropic: ModuleType, call: Callable[[], T]) -> T:
    try:
        return call()
    except anthropic.AuthenticationError:
        raise ValueError("API key Anthropic không hợp lệ") from None
    except anthropic.RateLimitError:
        raise ValueError("Anthropic báo vượt giới hạn; thử lại sau ít phút") from None
    except anthropic.APIStatusError as error:
        raise ValueError(f"Anthropic trả lỗi {error.status_code}: {error.message}") from None
    except anthropic.APIConnectionError:
        raise ValueError("không kết nối được tới Anthropic") from None


def models() -> dict:
    """Danh sách model của nhà cung cấp đang chọn.

    Returns:
        ``{"models": [id]}``.

    Raises:
        ValueError: Thiếu key, URL, hoặc nhà cung cấp trả lỗi.
    """
    current = _current()
    if current["provider"] == "anthropic":
        anthropic, client = _anthropic_client(current["api_key"])
        ids = _anthropic_errors(anthropic, lambda: [m.id for m in client.models.list()])
        return {"models": ids}
    data = _request(current, "/models", timeout=30)
    items = data.get("data", []) if isinstance(data, dict) else []
    return {"models": sorted({m["id"] for m in items if isinstance(m, dict) and m.get("id")})}


def _compact(value: object, depth: int = 0) -> object:
    """Rút gọn dữ kiện: danh sách dài cắt còn 12 mục, chuỗi dài cắt, lồng quá sâu bỏ."""
    if depth > 5:
        return "…"
    if isinstance(value, dict):
        return {
            str(k): _compact(v, depth + 1) for k, v in value.items() if v not in (None, [], {})
        }
    if isinstance(value, (list, tuple)):
        items = [_compact(v, depth + 1) for v in value[:12]]
        return items + ([f"… còn {len(value) - 12} mục"] if len(value) > 12 else [])
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "…"
    return value


def gather_facts(workdir: Workdir) -> dict:
    """Dữ kiện của Studio lúc hỏi: cảnh và split, approach, run gần đây, môi trường, GPU.

    Mục nào đọc lỗi thì ghi lỗi vào chỗ đó, không làm hỏng cả câu hỏi.
    """
    facts: dict[str, Any] = {}

    def section(name: str, read: Callable[[], object]) -> None:
        try:
            facts[name] = _compact(read())
        except Exception as error:  # noqa: BLE001 - dữ kiện thiếu một phần vẫn hỏi được
            facts[name] = f"không đọc được: {error}"

    def scene_items() -> list:
        items = []
        for manifest in scenes.list_scenes(workdir):
            item = manifest.model_dump(mode="json")
            item.pop("frames", None)
            splits = scenes.list_splits(workdir, manifest.scene_id)
            item["splits"] = [s.model_dump(mode="json") for s in splits]
            items.append(item)
        return items

    def approach_items() -> list:
        from drive2sim.engine.resolve import camera_support

        items = []
        for approach_id in list_approach_ids():
            approach = load_approach(approach_id)
            item = approach.model_dump(mode="json")
            item.pop("stages", None)
            item["camera_support"] = camera_support(approach)
            items.append(item)
        return items

    def run_items() -> list:
        keep = (
            "run_id", "scene_id", "split_id", "approach_id", "preset", "options", "status",
            "executor", "metrics", "quick", "created_at",
        )  # fmt: skip
        records = [r.model_dump(mode="json") for r in list_runs(workdir, None)[:15]]
        return [{k: r[k] for k in keep if k in r} for r in records]

    section("scenes", scene_items)
    section("approaches", approach_items)
    section("recent_runs", run_items)
    section("environment", lambda: environment_check("drivestudio", needs_smpl=True,
                                                     workdir_root=workdir.root))  # fmt: skip
    section("gpus", lambda: [{"name": n, "vram_mb": m} for n, m, _ in _gpus()])
    return facts


def parse_apply(text: str) -> Optional[dict]:
    """Khối ``d2s-apply`` cuối câu trả lời, chỉ giữ khoá biết; không có hoặc hỏng thì ``None``."""
    found = re.search(r"```d2s-apply\s*(\{.*?\})\s*```", text, re.S)
    if not found:
        return None
    try:
        data = json.loads(found.group(1))
    except ValueError:
        return None
    return {k: v for k, v in data.items() if k in APPLY_KEYS} if isinstance(data, dict) else None


def chat(messages: list, facts: dict, page: str = "") -> dict:
    """Hỏi trợ lý.

    Args:
        messages: ``[{"role": "user" | "assistant", "content"}]``, câu cuối của người dùng; giữ
            12 câu gần nhất.
        facts: Kết quả ``gather_facts``.
        page: Trang Studio người dùng đang mở.

    Returns:
        ``{"reply", "apply", "model"}``; ``apply`` theo ``parse_apply``.

    Raises:
        ValueError: Thiếu cài đặt, câu hỏi rỗng, nhà cung cấp lỗi hoặc từ chối trả lời.
    """
    current = _current()
    if not current["model"]:
        raise ValueError("chưa chọn model cho trợ lý (mở Cài đặt trong ngăn Trợ lý)")
    clean = [
        {"role": m["role"], "content": str(m["content"])[:6000]}
        for m in messages[-12:]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if not clean or clean[-1]["role"] != "user":
        raise ValueError("thiếu câu hỏi")
    context = "DỮ KIỆN STUDIO (JSON, lấy lúc hỏi):\n"
    context += json.dumps(facts, ensure_ascii=False, default=str)[:MAX_FACTS]
    if page:
        context += f"\nNgười dùng đang mở trang: {page}"
    system = SYSTEM + "\n\n" + context
    if current["provider"] == "anthropic":
        reply, model = _chat_anthropic(current, system, clean)
    else:
        payload = {
            "model": current["model"],
            "temperature": 0.3,
            # một system message: vài gateway chuẩn OpenAI chỉ giữ system message đầu tiên
            "messages": [{"role": "system", "content": system}, *clean],
        }
        data = _request(current, "/chat/completions", payload)
        try:
            reply = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"API trợ lý trả kết quả lạ: {str(data)[:300]}") from None
        model = data.get("model", current["model"])
    return {"reply": reply, "apply": parse_apply(reply), "model": model}


def _chat_anthropic(current: dict, system: str, messages: list) -> tuple[str, str]:
    anthropic, client = _anthropic_client(current["api_key"])
    model = current["model"]
    extra: dict[str, Any] = {}
    if model in CLAUDE_EFFORT:
        extra["output_config"] = {"effort": "low"}  # hỏi đáp ngắn, không cần nghĩ sâu
    if model in CLAUDE_FALLBACK:
        # bị từ chối thì API tự chạy lại trên model khác trong cùng lần gọi
        extra["betas"] = ["server-side-fallback-2026-07-01"]
        extra["fallbacks"] = "default"
        create = client.beta.messages.create
    else:
        create = client.messages.create
    response = _anthropic_errors(
        anthropic,
        lambda: create(model=model, max_tokens=16000, system=system, messages=messages, **extra),
    )
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise ValueError(
            f"Claude không trả lời câu này (refusal{f': {category}' if category else ''})"
        )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, getattr(response, "model", model)
