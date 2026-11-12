"""Route của nhóm approach: ``/api/approaches/...`` (CONTRACTS mục 11.1; SPEC mục 5 bước 4).

File này chứa:
- ``PlanBody``: thân yêu cầu chung của ``check`` và của ``POST /api/runs``.
- ``plan_from_body``: đọc cảnh, split, approach và lựa chọn rồi gọi ``resolve``, giống
  ``resolve_from_args`` của CLI.
- ``GET /api/approaches``, ``GET /api/approaches/{approach_id}``: đọc file approach.
- ``POST /api/approaches/{approach_id}/check``: kiểm tra trước khi chạy, như ``d2s approach
  check``: trả danh sách stage, stage nào dùng lại cache, và các lựa chọn bị khoá theo từng ô để
  giao diện làm mờ kèm lý do.
"""

from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from drive2sim.core.schema import OptionValue, Split
from drive2sim.engine.approaches import list_approach_ids, load_approach
from drive2sim.engine.resolve import Plan, blocked_choices, resolve
from drive2sim.engine.run import preview_reuse
from drive2sim.engine.stage import Scene
from drive2sim.studio.api.state import StudioState, state_of

router = APIRouter(prefix="/api/approaches", tags=["approaches"])


class PlanBody(BaseModel):
    """Lựa chọn của người dùng cho một lần chạy; cùng ý nghĩa với tuỳ chọn của ``d2s run start``.

    Attributes:
        scene_id: Cảnh.
        split_id: Split.
        preset: Preset, nếu có.
        options: Tên ô → lựa chọn, như ``--set <ô>=<giá trị>``.
        stage_params: ``stage_id`` → tham số, như ``--set <stage_id>.<tham số>=<giá trị>``.
    """

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    split_id: str
    preset: Optional[str] = None
    options: dict[str, OptionValue] = {}
    stage_params: dict[str, dict[str, Any]] = {}


def plan_from_body(
    state: StudioState, approach_id: str, body: PlanBody
) -> tuple[Scene, Split, Plan]:
    """Đọc cảnh, split, approach rồi quyết định danh sách stage.

    Args:
        state: Trạng thái của Studio.
        approach_id: Approach.
        body: Lựa chọn của người dùng.

    Returns:
        Bộ ``(cảnh, split, plan)``.

    Raises:
        WorkdirError: Nếu cảnh, split hoặc approach không có.
        ResolveError: Nếu preset, ô, giá trị hoặc tham số sai.
        BlockedError: Nếu dữ liệu hoặc tổ hợp lựa chọn không cho chạy.
    """
    scene = state.workdir.load_scene(body.scene_id)
    split = state.workdir.load_split(body.scene_id, body.split_id)
    plan = resolve(
        load_approach(approach_id),
        scene.manifest,
        split,
        preset=body.preset,
        option_values=body.options,
        stage_params=body.stage_params,
        registry=state.registry,
    )
    return scene, split, plan


@router.get("")
def list_approaches() -> dict:
    """Liệt kê các approach có sẵn.

    Returns:
        ``{"items": [approach, ...], "total": n}``.
    """
    items = [load_approach(a).model_dump(mode="json") for a in list_approach_ids()]
    return {"items": items, "total": len(items)}


@router.get("/{approach_id}")
def get_approach(approach_id: str) -> dict:
    """Trả về nội dung file approach: ô chọn, preset, ràng buộc, stage."""
    return load_approach(approach_id).model_dump(mode="json")


@router.post("/{approach_id}/check")
def check(approach_id: str, body: PlanBody, request: Request) -> dict:
    """Kiểm tra một approach chạy được trên cảnh và split không, không chạy gì.

    Returns:
        ``{"approach_id", "preset", "options", "lidar_usage", "stages": [{"stage_id",
        "params", "cache_hit"}], "blocked_choices": {ô: [{"choice", "message"}]}}``. Khi bị
        khoá thì trả lỗi ``422`` với các lý do (CONTRACTS mục 11.3).
    """
    state = state_of(request)
    scene, split, plan = plan_from_body(state, approach_id, body)
    reused = preview_reuse(plan, scene, split, state.workdir, state.registry)
    blocked = blocked_choices(load_approach(approach_id), scene.manifest, state.registry)
    return {
        "approach_id": plan.approach_id,
        "preset": plan.preset,
        "options": plan.options,
        "lidar_usage": plan.lidar_usage,
        "stages": [
            {"stage_id": stage.stage_id, "params": stage.params, "cache_hit": hit}
            for stage, hit in zip(plan.stages, reused)
        ],
        "blocked_choices": {
            option: [{"choice": r.choice, "message": r.message} for r in reasons]
            for option, reasons in blocked.items()
        },
    }
