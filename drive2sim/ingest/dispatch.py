"""Chọn dataset, phiên bản và chuỗi cần ingest, rồi gọi converter.

``d2s scene ingest`` và Studio cùng gọi file này, nên hai nơi chọn và báo lỗi giống nhau.

File này chứa:
- ``IngestRequestError``: lỗi do yêu cầu thiếu hoặc sai (mã lỗi ``invalid_request``).
- ``IngestTarget``: dataset, phiên bản, chuỗi đã chọn.
- ``plan_ingest``: từ kết quả ``inspect_source`` và lựa chọn của người dùng, chọn đúng một chuỗi;
  tự chọn khi chỉ có một khả năng, báo rõ các lựa chọn khi có nhiều.
- ``run_ingest``: gọi converter của dataset: ``ingest_drivestudio`` với tham số riêng của
  dataset (nuScenes: nội suy), hoặc ``ingest_kitti360`` (khung đầu, số khung, camera, khoảng hở).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from drive2sim.engine.workdir import Workdir
from drive2sim.ingest import kitti360
from drive2sim.ingest.detect import INGEST_SUPPORTED, SourceReport
from drive2sim.ingest.drivestudio import ingest_drivestudio


class IngestRequestError(ValueError):
    """Yêu cầu ingest thiếu hoặc sai: dataset không nhận ra, phiên bản hay chuỗi không có."""


@dataclass(frozen=True)
class IngestTarget:
    """Chuỗi đã chọn để ingest.

    Attributes:
        dataset: Khoá trong ``SOURCES`` của DriveStudio, ví dụ ``nuscenes``, hoặc ``kitti360``.
        version: Phiên bản hoặc split; chuỗi rỗng nếu dataset không chia phiên bản.
        sequence: Tên chuỗi trong nguồn.
    """

    dataset: str
    version: str
    sequence: str


def plan_ingest(
    report: SourceReport,
    source_type: Optional[str] = None,
    version: Optional[str] = None,
    sequence: Optional[str] = None,
) -> IngestTarget:
    """Chọn đúng một chuỗi để ingest.

    Args:
        report: Kết quả ``inspect_source`` của thư mục nguồn.
        source_type: Dataset người dùng chỉ định; mặc định theo nhận dạng.
        version: Phiên bản; tự chọn khi thư mục chỉ có một.
        sequence: Chuỗi; tự chọn khi phiên bản chỉ có một chuỗi.

    Returns:
        Chuỗi đã chọn.

    Raises:
        IngestRequestError: Nếu không nhận ra dataset, dataset chưa hỗ trợ ingest, hoặc phải
            chọn phiên bản, chuỗi mà người dùng chưa chọn hay chọn sai; thông báo liệt kê lựa
            chọn hợp lệ.
    """
    dataset = source_type or report.dataset
    if dataset is None:
        raise IngestRequestError(
            "không nhận ra dataset; chạy d2s source inspect để xem thư mục có gì"
        )
    if dataset not in INGEST_SUPPORTED:
        raise IngestRequestError(
            f"chưa hỗ trợ ingest {dataset}; hỗ trợ: {', '.join(sorted(INGEST_SUPPORTED))}"
        )
    versions = report.versions
    if version is None:
        if len(versions) != 1:
            raise IngestRequestError(
                f"chọn phiên bản trong: {', '.join(versions) or '(không có)'}"
            )
        version = next(iter(versions))
    if version not in versions:
        raise IngestRequestError(f"không có phiên bản {version}; có: {', '.join(versions)}")
    sequences = versions[version]
    if sequence is None:
        if len(sequences) != 1:
            shown = ", ".join(sequences[:10]) + (", ..." if len(sequences) > 10 else "")
            raise IngestRequestError(f"chọn một chuỗi; có {len(sequences)}: {shown}")
        sequence = sequences[0]
    if sequence not in sequences:
        raise IngestRequestError(f"không có chuỗi {sequence} trong thư mục nguồn")
    return IngestTarget(dataset, version, sequence)


def run_ingest(
    raw_dir: Path,
    workdir: Workdir,
    target: IngestTarget,
    scene_id: Optional[str] = None,
    interpolate_n: int = 4,
    workers: int = 4,
    kitti360_options: Optional[kitti360.Kitti360Options] = None,
) -> Path:
    """Ingest chuỗi đã chọn.

    Args:
        raw_dir: Thư mục dữ liệu gốc.
        workdir: Thư mục làm việc.
        target: Chuỗi đã chọn bằng ``plan_ingest``.
        scene_id: Định danh cảnh trong tool; mặc định ``<dataset>_<chuỗi>``.
        interpolate_n: nuScenes: số khung nội suy giữa hai keyframe 2 Hz; ``4`` cho 10 Hz như
            teammate đang dùng, ``0`` để giữ 2 Hz. Dataset khác bỏ qua.
        workers: Số tiến trình song song của script.
        kitti360_options: KITTI-360: khung đầu, số khung, camera, khoảng hở tối đa; mặc định
            ``Kitti360Options()``. Dataset khác bỏ qua.

    Returns:
        Thư mục cảnh ``data/<scene_id>/``.

    Raises:
        IngestError: Nếu ingest lỗi (xem ``ingest_drivestudio``, ``ingest_kitti360``).
    """
    if target.dataset == kitti360.DATASET:
        return kitti360.ingest_kitti360(
            raw_dir, workdir, target.sequence, scene_id=scene_id, options=kitti360_options
        )
    extra = ["--interpolate_N", str(interpolate_n)] if target.dataset == "nuscenes" else []
    return ingest_drivestudio(
        raw_dir,
        workdir,
        target.dataset,
        target.sequence,
        version=target.version or None,
        scene_id=scene_id,
        workers=workers,
        extra_args=extra,
    )
