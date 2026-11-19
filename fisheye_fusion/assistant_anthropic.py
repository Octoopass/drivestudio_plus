"""Anthropic (Claude) provider for the in-app assistant, through the official `anthropic` SDK.

Kept separate from the OpenAI-compatible path in assistant.py. Notes for the current models:
- no `temperature` (rejected on current models) and no `thinking` parameter (adaptive thinking is the default
  and the only mode on Claude Opus 5.5); only text blocks are read from the reply;
- on models that support it, a refusal is retried server-side on a fallback model (`fallbacks: "default"`);
- `max_tokens` 16000 keeps a non-streaming request well inside the SDK's HTTP timeout.
"""
from __future__ import annotations

DEFAULT_MODEL = "claude-opus-5-5"
# models that accept the server-side refusal fallback ("default" routing by refusal category)
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key, timeout=180.0, max_retries=2)


def _error(exc) -> ValueError:
    import anthropic
    if isinstance(exc, anthropic.AuthenticationError):
        return ValueError("Anthropic: API key không hợp lệ")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return ValueError("Anthropic: key không có quyền dùng model này")
    if isinstance(exc, anthropic.NotFoundError):
        return ValueError("Anthropic: không có model này")
    if isinstance(exc, anthropic.RateLimitError):
        return ValueError("Anthropic: vượt giới hạn tốc độ, thử lại sau ít phút")
    if isinstance(exc, anthropic.APIStatusError):
        return ValueError(f"Anthropic trả lỗi {exc.status_code}: {str(exc.message)[:300]}")
    if isinstance(exc, anthropic.APIConnectionError):
        return ValueError("Không kết nối được tới Anthropic API")
    return ValueError(f"Anthropic lỗi: {exc}")


def list_models(api_key: str) -> list[str]:
    import anthropic
    try:
        return sorted(m.id for m in _client(api_key).models.list(limit=100))
    except anthropic.AnthropicError as exc:
        raise _error(exc) from exc


def chat(api_key: str, model: str, system: str, messages: list[dict]) -> dict:
    """messages: [{role: user|assistant, content: str}], starting with a user turn."""
    import anthropic
    model = model or DEFAULT_MODEL
    try:
        if model in FALLBACK_MODELS:
            resp = _client(api_key).beta.messages.create(
                model=model, max_tokens=16000, system=system, messages=messages,
                betas=[FALLBACK_BETA], fallbacks="default",
            )
        else:
            resp = _client(api_key).messages.create(model=model, max_tokens=16000, system=system, messages=messages)
    except anthropic.AnthropicError as exc:
        raise _error(exc) from exc
    if resp.stop_reason == "refusal":
        detail = getattr(getattr(resp, "stop_details", None), "explanation", None)
        return {"reply": "Model từ chối trả lời câu hỏi này" + (f": {detail}" if detail else "."), "model": resp.model}
    text = "".join(b.text for b in resp.content if b.type == "text")
    if resp.stop_reason == "max_tokens":
        text += "\n\n(Câu trả lời bị cắt vì quá dài.)"
    return {"reply": text, "model": resp.model}
