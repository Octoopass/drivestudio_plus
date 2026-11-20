"""Route trợ lý: ``/api/assistant/...`` (CONTRACTS mục 11.1; ``studio/api/assistant.py``).

File này chứa:
- ``GET /api/assistant/config``, ``POST /api/assistant/config``: nhà cung cấp, URL, model, key
  (key không bao giờ trả về).
- ``GET /api/assistant/models``: model của nhà cung cấp đang chọn.
- ``POST /api/assistant/chat``: hỏi kèm dữ kiện của Studio lúc hỏi.
"""

from typing import Literal, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.studio.api import assistant
from drive2sim.studio.api.state import state_of

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class ConfigBody(BaseModel):
    """Thân của ``POST /api/assistant/config``; trường bỏ trống thì giữ nguyên.

    Attributes:
        provider: Nhà cung cấp.
        base_url: URL của endpoint tự đặt (``custom``).
        model: Model của nhà cung cấp đang chọn.
        api_key: Key mới của nhà cung cấp đang chọn.
        clear_key: Xoá key của nhà cung cấp đang chọn.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Optional[str] = None
    base_url: Optional[str] = Field(None, max_length=300)
    model: Optional[str] = Field(None, max_length=120)
    api_key: Optional[str] = Field(None, max_length=500)
    clear_key: bool = False


class Message(BaseModel):
    """Một câu trong cuộc hỏi đáp."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class ChatBody(BaseModel):
    """Thân của ``POST /api/assistant/chat``.

    Attributes:
        messages: Cuộc hỏi đáp, câu cuối là câu hỏi.
        page: Trang Studio đang mở.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1, max_length=40)
    page: str = Field("", max_length=200)


@router.get("/config")
def get_config() -> dict:
    """Cài đặt trợ lý (không có key): nhà cung cấp, URL, model, ``has_key``, ``key_hint``."""
    return assistant.config()


@router.post("/config")
def set_config(body: ConfigBody) -> dict:
    """Đổi cài đặt; trả cài đặt mới (không có key)."""
    return assistant.set_config(
        body.provider, body.base_url, body.model, body.api_key, body.clear_key
    )


@router.get("/models")
def list_models() -> dict:
    """Model của nhà cung cấp đang chọn: ``{"models": [id]}``."""
    return assistant.models()


@router.post("/chat")
def chat(body: ChatBody, request: Request) -> dict:
    """Hỏi trợ lý; trả ``{"reply", "apply", "model"}``."""
    facts = assistant.gather_facts(state_of(request).workdir)
    messages = [m.model_dump() for m in body.messages]
    return assistant.chat(messages, facts, body.page)
