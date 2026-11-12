"""Định nghĩa khuôn mẫu mà mọi stage phải theo, và những gì engine đưa cho stage khi chạy.

Stage là một bước xử lý trong pipeline của approach, ví dụ sinh mask, khởi tạo điểm, train,
đánh giá (SPEC mục 3). File này chứa:
- ``Stage``: lớp gốc của mọi stage. Mỗi stage mới là một lớp con, khai báo stage cần đầu vào
  gì, tạo ra đầu ra gì, có tham số gì, rồi cài hàm ``run`` làm việc thật.
- ``StageContext``: những thứ engine đưa cho stage lúc chạy: thư mục ghi kết quả, cảnh, split,
  và hai hàm báo tiến độ, báo metric.
- ``Scene``: một cảnh đã ingest, gồm thư mục dữ liệu chuẩn và manifest.
- ``NoParams``: lớp tham số rỗng cho stage không có tham số.

Stage chỉ làm đúng một việc: đọc đầu vào, ghi file vào ``ctx.output_dir`` và trả về đường dẫn
của đầu ra. Những việc chung như cache, file ``done``, thư mục ``.partial``, ``artifact.json``
do engine lo (``drive2sim/engine/run.py``), nên người viết stage không phải lặp lại.

Người viết stage mới đọc file này cùng CONTRACTS mục 5.1; stage được đăng ký bằng
``@register_stage`` trong ``drive2sim/engine/registry.py``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

from drive2sim.core.schema import Manifest, Split

# Hai tên đặc biệt dùng trong ``Stage.inputs``: không phải artifact mà là cảnh và split của run.
SCENE_INPUT = "scene"
SPLIT_INPUT = "split"


@dataclass(frozen=True)
class Scene:
    """Một cảnh đã được ingest: thư mục dữ liệu chuẩn cùng manifest của nó.

    Engine tạo đối tượng này một lần cho mỗi run (qua ``Workdir.load_scene``) rồi đưa cho mọi
    stage qua ``ctx.scene``. Stage dùng nó để biết cảnh có những camera nào, có LiDAR không, và
    tìm tới thư mục dữ liệu chuẩn để đọc (CONTRACTS mục 2).

    Attributes:
        scene_id: Định danh cảnh, ví dụ ``nuscenes_scene-0061``.
        root: Thư mục ``data/<scene_id>/`` của cảnh.
        manifest: Nội dung ``manifest.yaml`` đã kiểm tra theo schema.
    """

    scene_id: str
    root: Path
    manifest: Manifest


class StageContext(Protocol):
    """Những gì engine đưa cho một stage khi gọi ``Stage.run``.

    Đây là một interface: engine có cách cài riêng (``drive2sim/engine/run.py``), test có thể
    dùng cách cài khác. Stage chỉ được ghi file vào ``output_dir``; muốn báo tiến độ hoặc giá trị
    metric thì gọi hai phương thức bên dưới, engine sẽ chuyển thành sự kiện cho CLI, Studio hoặc
    worker (CONTRACTS mục 9.1, 11.4).

    Attributes:
        output_dir: Thư mục stage được ghi kết quả vào; engine đã tạo sẵn và đang trống.
        scene: Cảnh của run.
        split: Split train/test của run.
    """

    output_dir: Path
    scene: Scene
    split: Split

    def progress(self, fraction: float, message: str = "") -> None:
        """Báo tiến độ của stage.

        Args:
            fraction: Phần đã xong, từ 0 đến 1; giá trị ngoài khoảng này bị kẹp lại.
            message: Mô tả ngắn bằng tiếng Việt về việc đang làm, ví dụ ``"đang tính mask"``.
        """

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Báo một giá trị metric trong lúc chạy, ví dụ loss khi train.

        Args:
            name: Tên metric, ``snake_case``, ví dụ ``"loss"``.
            value: Giá trị.
            step: Bước (iteration) ứng với giá trị này.
        """


class NoParams(BaseModel):
    """Lớp tham số cho stage không có tham số nào.

    Từ chối mọi khoá, để lỗi gõ sai tham số trong file approach hoặc ``--set`` được báo ngay
    thay vì bị bỏ qua.
    """

    model_config = ConfigDict(extra="forbid")


class Stage(ABC):
    """Lớp gốc của mọi stage.

    Muốn thêm một stage, tạo một lớp con trong ``drive2sim/stages/<nhóm>/<tên>.py``, khai báo các
    thuộc tính lớp bên dưới, cài ``run``, rồi đặt ``@register_stage("<nhóm>.<tên>")`` trên lớp.
    Decorator tự gán ``stage_id`` và kiểm tra khai báo ngay khi import, nên khai báo sai bị báo
    trước khi run bắt đầu (``drive2sim/engine/registry.py``).

    Ví dụ khai báo một stage khởi tạo điểm từ depth::

        @register_stage("init.depth")
        class InitDepthStage(Stage):
            version = "0.1.0"
            cacheable = True
            inputs = {"scene": "scene", "depth": "depth_prior"}
            outputs = {"points": "points_init"}
            Params = InitDepthParams

            def run(self, ctx, inputs, params):
                ...  # đọc inputs["depth"], ghi ctx.output_dir / "points_init.npz"
                return {"points": ctx.output_dir / "points_init.npz"}

    Attributes:
        stage_id: ``<nhóm>.<tên>``, ví dụ ``init.depth``; do decorator gán, không tự đặt.
        version: Phiên bản ``MAJOR.MINOR.PATCH``. Phải tăng mỗi khi đổi hành vi của stage, vì
            phiên bản nằm trong khoá cache: quên tăng thì kết quả cũ vẫn bị dùng lại.
        cacheable: ``True`` với nhóm ``prep`` và ``init`` (kết quả vào ``cache/``, dùng chung
            giữa các approach), ``False`` với các nhóm còn lại (kết quả vào thư mục run).
            Xem CONTRACTS mục 5.2.
        uses_lidar: ``True`` nếu stage cần LiDAR. Engine khoá stage này khi cảnh không có
            LiDAR, và ghi tên stage vào ``lidar_usage`` của ``run.json``.
        inputs: Tên đầu vào → loại artifact (bảng CONTRACTS mục 5.4), hoặc ``"scene"`` /
            ``"split"`` khi kết quả phụ thuộc cảnh hoặc split; khai báo hai tên này để khoá
            cache đổi khi cảnh hoặc split đổi.
        outputs: Tên đầu ra → loại artifact; đúng một mục.
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
        """Chạy stage: đọc đầu vào, ghi kết quả vào ``ctx.output_dir``.

        Engine gọi hàm này đúng một lần cho mỗi lần stage thật sự chạy; khi kết quả đã có sẵn
        (trong cache hoặc có file ``done``) thì không gọi.

        Args:
            ctx: Ngữ cảnh do engine tạo; cảnh và split đọc từ ``ctx.scene``, ``ctx.split``.
            inputs: Tên đầu vào artifact → thư mục chứa artifact đó (thư mục kết quả của stage
                phía trước). Không có ``"scene"`` và ``"split"``; hai thứ này đọc từ ``ctx``.
            params: Tham số đã gộp và đã kiểm tra theo ``Params``.

        Returns:
            Tên đầu ra → đường dẫn tới file hoặc thư mục kết quả, nằm trong ``ctx.output_dir``.
            Phải có đúng các tên trong ``outputs``, ví dụ
            ``{"points": ctx.output_dir / "points_init.npz"}``.
        """
