"""Quản lý cache cho kết quả của các stage dùng chung giữa các approach.

Stage nhóm ``prep`` và ``init`` (mask, SMPL, depth, điểm khởi tạo...) thường tốn thời gian và cho
cùng kết quả với mọi approach trên cùng cảnh. Kết quả của chúng được lưu ở
``cache/<scene_id>/<stage_id>@<version>-<hash8>/`` để approach sau dùng lại, không phải chạy lại
(ARCHITECTURE mục 2.7, CONTRACTS mục 5.2).

File này chứa:
- ``CacheStore``: tìm thư mục cache đã có (``lookup``), tạo thư mục tạm ``.partial`` để ghi
  (``begin``), đổi tên thành thư mục chính thức khi ghi xong (``commit``), hoặc xoá khi stage
  lỗi (``abort``).
- ``input_key_for_scene``, ``input_key_for_split``, ``input_key_for_run_stage``: tạo khoá đại
  diện cho từng loại đầu vào, để khoá cache đổi khi đầu vào đổi.
- ``CacheBusyError``: lỗi khi một tiến trình khác đang ghi cùng khoá.

Cách bảo đảm an toàn: thư mục đang ghi mang hậu tố ``.partial`` và chỉ đổi tên khi ghi xong, nên
thư mục cache không có ``.partial`` luôn là kết quả đầy đủ, kể cả khi tiến trình bị ngắt giữa
chừng. Thư mục ``.partial`` cũng là khoá: chỉ một tiến trình tạo được nó, nên hai tiến trình
không ghi cùng một khoá cùng lúc.

``run.py`` dùng file này khi chạy một stage có ``cacheable = True``.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from drive2sim.core import identifiers

PARTIAL_SUFFIX = ".partial"
OWNER_FILE = ".owner"  # pid của tiến trình đang ghi, đặt trong thư mục .partial
ARTIFACT_FILE = "artifact.json"


class CacheBusyError(RuntimeError):
    """Lỗi khi một tiến trình khác còn sống đang ghi cùng khoá cache.

    Ứng với mã lỗi ``conflict`` (CONTRACTS mục 11.3). Người dùng chờ tiến trình kia xong rồi
    chạy lại; khi đó kết quả đã có trong cache và được dùng lại.
    """


def input_key_for_scene(scene_id: str, converter_version: str) -> str:
    """Tạo khoá đại diện cho dữ liệu chuẩn của một cảnh, dùng khi stage nhận đầu vào ``"scene"``.

    Khoá gồm cả phiên bản converter: nếu cảnh được ingest lại bằng converter khác, khoá đổi và
    cache cũ không bị dùng nhầm (ARCHITECTURE mục 2.7).

    Args:
        scene_id: Định danh cảnh.
        converter_version: ``source.converter_version`` trong manifest.

    Returns:
        Khoá dạng ``"scene:<scene_id>@<converter_version>"``, ví dụ
        ``"scene:nuscenes_scene-0061@19.9.0"``.
    """
    return f"scene:{scene_id}@{converter_version}"


def input_key_for_split(split_id: str) -> str:
    """Tạo khoá đại diện cho một split, dùng khi stage nhận đầu vào ``"split"``.

    Args:
        split_id: Định danh split.

    Returns:
        Khoá dạng ``"split:<split_id>"``, ví dụ ``"split:every8_allcams"``.
    """
    return f"split:{split_id}"


def input_key_for_run_stage(run_id: str, stage_id: str) -> str:
    """Tạo khoá đại diện cho kết quả của một stage không cache trong một run.

    Kết quả của stage không cache (``train.*``, ``eval.*``...) chỉ thuộc về một run, nên khoá
    gồm ``run_id``. Khoá này được ghi vào ``artifact.json`` của stage phía sau dùng nó.

    Args:
        run_id: Định danh run.
        stage_id: Định danh stage.

    Returns:
        Khoá dạng ``"run:<run_id>/<stage_id>"``.
    """
    return f"run:{run_id}/{stage_id}"


def _pid_alive(pid: int) -> bool:
    """Kiểm tra tiến trình ``pid`` còn chạy trên máy này không.

    Gửi tín hiệu 0 tới tiến trình: không làm gì tiến trình nhưng báo lỗi nếu nó không tồn tại.

    Args:
        pid: Mã tiến trình.

    Returns:
        ``True`` nếu tiến trình còn sống (kể cả khi thuộc người dùng khác), ``False`` nếu không.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class CacheStore:
    """Thư mục ``cache/`` của một thư mục làm việc, cùng các thao tác đọc ghi an toàn.

    Thứ tự dùng khi chạy một stage có cache: gọi ``lookup``; nếu chưa có thì ``begin`` để lấy
    thư mục ``.partial``, chạy stage ghi vào đó, rồi ``commit`` khi xong hoặc ``abort`` khi lỗi.

    Attributes:
        root: Thư mục ``cache/``.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, cache_key: str) -> Path:
        """Trả về thư mục chính thức của một khoá cache (có thể chưa tồn tại).

        Args:
            cache_key: Khoá dạng ``<scene_id>/<stage_id>@<version>-<hash8>``.

        Returns:
            Đường dẫn ``cache/<cache_key>/``.

        Raises:
            ValueError: Nếu khoá sai định dạng.
        """
        identifiers.check_cache_key(cache_key)
        return self.root / cache_key

    def lookup(self, cache_key: str) -> Optional[Path]:
        """Tìm kết quả đã có của một khoá cache.

        Kết quả được coi là đầy đủ khi thư mục chính thức có ``artifact.json`` (engine chỉ ghi
        file này sau khi stage chạy xong).

        Args:
            cache_key: Khoá cache.

        Returns:
            Thư mục kết quả nếu đã có và đầy đủ, ``None`` nếu chưa có.
        """
        path = self.path(cache_key)
        return path if (path / ARTIFACT_FILE).is_file() else None

    def begin(self, cache_key: str) -> Path:
        """Tạo thư mục tạm ``.partial`` để ghi kết quả của một khoá cache.

        ``mkdir`` là thao tác nguyên tử, nên nếu hai tiến trình cùng gọi thì chỉ một tiến trình
        tạo được thư mục; tiến trình đó ghi pid của mình vào ``.owner``. Nếu thư mục ``.partial``
        đã có nhưng tiến trình ghi nó đã chết (ví dụ máy tắt giữa chừng), thư mục bị xoá và tạo
        lại.

        Args:
            cache_key: Khoá cache.

        Returns:
            Thư mục ``.partial`` rỗng (chỉ có file ``.owner``) để stage ghi kết quả vào.

        Raises:
            CacheBusyError: Nếu một tiến trình khác còn sống đang ghi khoá này.
        """
        final = self.path(cache_key)
        partial = final.with_name(final.name + PARTIAL_SUFFIX)
        partial.parent.mkdir(parents=True, exist_ok=True)
        try:
            partial.mkdir()
        except FileExistsError:
            owner = partial / OWNER_FILE
            pid = int(owner.read_text()) if owner.is_file() else None
            if pid is not None and pid != os.getpid() and _pid_alive(pid):
                raise CacheBusyError(
                    f"tiến trình {pid} đang ghi cache {cache_key}; chờ nó xong rồi chạy lại"
                ) from None
            shutil.rmtree(partial)
            partial.mkdir()
        (partial / OWNER_FILE).write_text(str(os.getpid()))
        return partial

    def commit(self, partial: Path) -> Path:
        """Biến thư mục ``.partial`` đã ghi xong thành thư mục cache chính thức.

        Xoá ``.owner`` rồi đổi tên thư mục; đổi tên trên cùng ổ đĩa là thao tác nguyên tử. Nếu
        thư mục chính thức đã có (một lần chạy khác vừa ghi xong cùng khoá), giữ bản đã có và
        xoá bản ``.partial``, vì hai bản cùng khoá cho cùng kết quả.

        Args:
            partial: Thư mục ``.partial`` do ``begin`` trả về.

        Returns:
            Thư mục cache chính thức.
        """
        final = partial.with_name(partial.name[: -len(PARTIAL_SUFFIX)])
        (partial / OWNER_FILE).unlink(missing_ok=True)
        if final.exists():
            shutil.rmtree(partial)
            return final
        os.replace(partial, final)
        return final

    def abort(self, partial: Path) -> None:
        """Xoá thư mục ``.partial`` khi stage lỗi hoặc bị ngắt, để không để lại kết quả dở.

        Args:
            partial: Thư mục ``.partial`` do ``begin`` trả về.
        """
        shutil.rmtree(partial, ignore_errors=True)
