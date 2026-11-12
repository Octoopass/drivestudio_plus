"""Interface mà mọi stage phải theo (CONTRACTS mục 5.1).

Stage chỉ làm một việc: đọc đầu vào, ghi file vào ``ctx.output_dir`` và trả về đường dẫn của
từng đầu ra. Cache, file ``done``, thư mục ``.partial`` và ``artifact.json`` do engine lo.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

from drive2sim.core.schema import Manifest, Split

# Tên đặc biệt trong ``Stage.inputs``: không phải artifact mà là cảnh và split của run.
SCENE_INPUT = "scene"
SPLIT_INPUT = "split"


@dataclass(frozen=True)
class Scene:
    """Một cảnh đã ingest: thư mục dữ liệu chuẩn và manifest của nó (CONTRACTS mục 2).

    Attributes:
        scene_id: Định danh cảnh.
        root: Thư mục ``data/<scene_id>/``.
        manifest: Nội dung ``manifest.yaml`` đã kiểm tra.
    """

    scene_id: str
    root: Path
    manifest: Manifest

    @property
    def ncore_dir(self) -> Path:
        """Thư mục kho NCore V4 của cảnh."""
        return self.root / "ncore"


class StageContext(Protocol):
    """Những gì engine đưa cho stage khi chạy.

    Stage chỉ ghi vào ``output_dir``; báo tiến độ và metric qua hai phương thức bên dưới để
    engine chuyển tới CLI, Studio hoặc worker (CONTRACTS mục 9.1, 11.4).
    """

    output_dir: Path
    scene: Scene
    split: Split

    def progress(self, fraction: float, message: str = "") -> None:
        """Báo tiến độ trong stage, ``fraction`` từ 0 đến 1."""

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Báo một giá trị metric, ví dụ loss khi train."""


class NoParams(BaseModel):
    """Tham số mặc định của stage không có tham số nào; từ chối mọi khoá."""

    model_config = ConfigDict(extra="forbid")


class Stage(ABC):
    """Lớp gốc của mọi stage.

    Lớp con khai báo các thuộc tính lớp dưới đây và cài ``run``, rồi đăng ký bằng
    ``@register_stage("<nhóm>.<tên>")``; decorator gán ``stage_id`` và kiểm tra khai báo
    (``drive2sim/engine/registry.py``).

    Attributes:
        stage_id: ``<nhóm>.<tên>``, do decorator gán.
        version: ``MAJOR.MINOR.PATCH``; tăng khi đổi hành vi, vì nó nằm trong khoá cache.
        cacheable: ``True`` với nhóm ``prep`` và ``init`` (kết quả vào ``cache/``), ``False``
            với các nhóm còn lại (kết quả vào thư mục run). CONTRACTS mục 5.2.
        uses_lidar: Stage có đọc LiDAR không; engine ghi vào ``run.json``.
        inputs: Tên đầu vào → loại artifact, hoặc ``"scene"`` / ``"split"`` khi kết quả phụ
            thuộc cảnh hoặc split (để khoá cache đổi theo chúng).
        outputs: Tên đầu ra → loại artifact.
        Params: Lớp Pydantic mô tả tham số; nên đặt ``extra="forbid"`` như ``NoParams``.
    """

    stage_id: ClassVar[str] = ""
    version: ClassVar[str] = ""
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {}
    Params: ClassVar[type[BaseModel]] = NoParams

    @abstractmethod
    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy stage.

        Args:
            ctx: Ngữ cảnh do engine tạo; cảnh và split lấy từ đây.
            inputs: Tên đầu vào artifact → thư mục của artifact đó. Không có ``"scene"`` và
                ``"split"``; hai thứ này đọc từ ``ctx``.
            params: Tham số đã gộp và đã kiểm tra theo ``Params``.

        Returns:
            Tên đầu ra → đường dẫn nằm trong ``ctx.output_dir``; phải có đủ mọi tên trong
            ``outputs``.
        """
