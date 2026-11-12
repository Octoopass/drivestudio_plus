"""Quản lý đường dẫn trong thư mục làm việc của tool.

Thư mục làm việc (biến môi trường ``D2S_WORKDIR``) chứa ba thư mục con: ``data/`` (dữ liệu
chuẩn của các cảnh), ``cache/`` (kết quả stage dùng chung) và ``runs/`` (kết quả từng run). Cấu
trúc bên trong theo CONTRACTS mục 2.1, 5.2 và 6.1.

File này chứa:
- ``Workdir``: biết mọi đường dẫn trong thư mục làm việc (thư mục cảnh, file split, thư mục
  run), đọc manifest và split, tìm thư mục của một run theo ``run_id``.
- ``WorkdirError``: lỗi khi không tìm thấy cảnh, split hoặc run.

Mọi code khác (engine, CLI, Studio) lấy đường dẫn qua ``Workdir`` thay vì tự ghép chuỗi, để cấu
trúc thư mục chỉ được viết ở một chỗ; đổi cấu trúc thì chỉ sửa file này.
"""

import os
from pathlib import Path
from typing import Optional

from drive2sim.core import files
from drive2sim.core.schema import Manifest, Split
from drive2sim.engine.stage import Scene

WORKDIR_ENV = "D2S_WORKDIR"


class WorkdirError(LookupError):
    """Lỗi khi không tìm thấy cảnh, split hoặc run trong thư mục làm việc.

    Ứng với mã lỗi ``not_found`` (CONTRACTS mục 11.3). Thông báo bằng tiếng Việt và gợi ý lệnh
    cần chạy, ví dụ ``d2s scene ingest`` khi cảnh chưa có.
    """


class Workdir:
    """Thư mục làm việc của tool và mọi đường dẫn bên trong nó.

    Tạo bằng ``Workdir.from_env()`` để theo đúng thứ tự ưu tiên của CLI, hoặc ``Workdir(path)``
    khi đã biết thư mục (ví dụ trong test).

    Attributes:
        root: Đường dẫn tuyệt đối của thư mục làm việc.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    @classmethod
    def from_env(cls, override: Optional[Path] = None) -> "Workdir":
        """Chọn thư mục làm việc theo thứ tự ưu tiên của CONTRACTS mục 10.4.

        Thứ tự: ``override`` (tuỳ chọn ``--workdir`` của CLI), rồi biến môi trường
        ``D2S_WORKDIR``, rồi thư mục hiện tại.

        Args:
            override: Thư mục do người dùng chỉ định bằng ``--workdir``, hoặc ``None``.

        Returns:
            Thư mục làm việc đã chọn.
        """
        if override is not None:
            return cls(override)
        return cls(Path(os.environ.get(WORKDIR_ENV, os.getcwd())))

    @property
    def data_dir(self) -> Path:
        """Trả về thư mục ``data/`` chứa dữ liệu chuẩn của mọi cảnh."""
        return self.root / "data"

    @property
    def cache_dir(self) -> Path:
        """Trả về thư mục ``cache/`` chứa kết quả của các stage dùng chung."""
        return self.root / "cache"

    @property
    def runs_dir(self) -> Path:
        """Trả về thư mục ``runs/`` chứa kết quả của mọi run."""
        return self.root / "runs"

    def scene_dir(self, scene_id: str) -> Path:
        """Trả về thư mục ``data/<scene_id>/`` của một cảnh (có thể chưa tồn tại).

        Args:
            scene_id: Định danh cảnh, ví dụ ``nuscenes_scene-0061``.

        Returns:
            Đường dẫn thư mục cảnh.
        """
        return self.data_dir / scene_id

    def split_path(self, scene_id: str, split_id: str) -> Path:
        """Trả về đường dẫn file split ``data/<scene_id>/splits/<split_id>.json``.

        Args:
            scene_id: Định danh cảnh.
            split_id: Định danh split, ví dụ ``every8_allcams``.

        Returns:
            Đường dẫn file split (có thể chưa tồn tại).
        """
        return self.scene_dir(scene_id) / "splits" / f"{split_id}.json"

    def run_dir(self, scene_id: str, approach_id: str, run_id: str) -> Path:
        """Trả về thư mục ``runs/<scene_id>/<approach_id>/<run_id>/`` của một run.

        Cấu trúc này trùng với cách DriveStudio đặt thư mục log
        (``output_root/project/run_name``), nên log của DriveStudio và file của tool nằm chung
        một thư mục run (CONTRACTS mục 6.1).

        Args:
            scene_id: Định danh cảnh.
            approach_id: Định danh approach.
            run_id: Định danh run.

        Returns:
            Đường dẫn thư mục run (có thể chưa tồn tại).
        """
        return self.runs_dir / scene_id / approach_id / run_id

    def relative(self, path: Path) -> str:
        """Đổi một đường dẫn thành dạng tương đối với thư mục làm việc.

        File JSON/YAML do tool ghi chỉ chứa đường dẫn tương đối, để chép cả thư mục làm việc
        sang máy khác vẫn dùng được (CONTRACTS mục 10.3).

        Args:
            path: Đường dẫn nằm trong thư mục làm việc.

        Returns:
            Đường dẫn tương đối, dấu ``/``, ví dụ ``"runs/nuscenes_scene-0061/omnire/..."``.

        Raises:
            ValueError: Nếu ``path`` nằm ngoài thư mục làm việc.
        """
        return Path(path).resolve().relative_to(self.root).as_posix()

    def load_scene(self, scene_id: str) -> Scene:
        """Đọc manifest của một cảnh đã ingest.

        Args:
            scene_id: Định danh cảnh.

        Returns:
            ``Scene`` gồm thư mục cảnh và manifest đã kiểm tra.

        Raises:
            WorkdirError: Nếu cảnh chưa được ingest (không có ``manifest.yaml``).
            ValueError: Nếu ``manifest.yaml`` sai schema.
        """
        manifest_path = self.scene_dir(scene_id) / "manifest.yaml"
        if not manifest_path.is_file():
            raise WorkdirError(
                f"không tìm thấy cảnh {scene_id}: thiếu {self.relative(manifest_path)}; "
                "chạy d2s scene ingest trước"
            )
        manifest = files.load(manifest_path, Manifest)
        return Scene(scene_id, self.scene_dir(scene_id), manifest)

    def load_split(self, scene_id: str, split_id: str) -> Split:
        """Đọc một split của cảnh.

        Args:
            scene_id: Định danh cảnh.
            split_id: Định danh split.

        Returns:
            Split đã kiểm tra theo schema.

        Raises:
            WorkdirError: Nếu file split không tồn tại.
            ValueError: Nếu file split sai schema.
        """
        path = self.split_path(scene_id, split_id)
        if not path.is_file():
            raise WorkdirError(
                f"không tìm thấy split {split_id} của cảnh {scene_id}; tạo bằng d2s split create"
            )
        return files.load(path, Split)

    def find_run(self, run_id: str) -> Path:
        """Tìm thư mục của một run khi chỉ biết ``run_id``.

        ``run_id`` có dạng ``YYYYMMDD-HHMMSS-<approach_id>-<hex>``, chứa ``approach_id`` nhưng
        không chứa ``scene_id``, nên hàm tìm trong mọi cảnh. Dùng cho các lệnh CLI chỉ nhận
        ``run_id`` như ``d2s run resume <run_id>``.

        Args:
            run_id: Định danh run.

        Returns:
            Thư mục run.

        Raises:
            WorkdirError: Nếu không có run nào mang ``run_id`` này.
        """
        approach_id = run_id.split("-")[2] if run_id.count("-") >= 3 else "*"
        matches = sorted(self.runs_dir.glob(f"*/{approach_id}/{run_id}"))
        if not matches:
            raise WorkdirError(f"không tìm thấy run {run_id} trong {self.runs_dir}")
        return matches[0]
