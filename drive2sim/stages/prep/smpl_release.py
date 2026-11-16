"""Stage ``prep.smpl_release``: lấy SMPL người đi bộ mà DriveStudio phát hành sẵn thay vì tự tính.

DriveStudio phát hành ``humanpose/smpl.pkl`` đã tính sẵn cho một số cảnh (``docs/datasets/
human_pose.md``, "Option 1"). Với cảnh có trong bản phát hành, stage này tải file zip (một lần mỗi
máy, giữ ở ``~/.cache/drive2sim/smpl_release/``), kiểm mã băm, lấy đúng file của cảnh rồi ghi lại
thành ``humanpose/smpl.pkl`` của tool, cùng chỗ và cùng loại artifact ``smpl`` như ``prep.smpl``,
nên ``train.drivestudio`` nhận như nhau. Không tốn GPU; ``prep.smpl`` (4D-Humans + PHALP) mất
khoảng 72 phút trên T4 với nuScenes ``scene-0061``.

So với ``prep.smpl`` trên nuScenes ``scene-0061`` (đo 16/11/2026): cùng 19 người, mọi mặt nạ khung
và camera trùng khớp, tham số tư thế lệch trung bình 1e-4 (lớn nhất 0,008, khoảng 0,4°), tham số
dáng người lệch trung bình 5e-4. Cùng phương pháp, nên kết quả gần như một.

File phát hành là pickle tải từ Internet: đọc bằng ``drive2sim/core/safe_pickle.py`` (chỉ cho phép
dict và tensor), rồi ghi lại bằng ``joblib`` như ``prep.smpl``; trainer chỉ nạp file do tool ghi.

Cảnh không có trong bản phát hành thì ``unavailable_reason`` báo lý do: engine khoá lựa chọn này và
mục cấu hình ``smpl: auto`` của approach chọn ``prep.smpl``.

File này chứa ``SmplRelease``, ``RELEASES``, ``release_member``, ``PrepSmplReleaseStage``.
"""

import hashlib
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel

from drive2sim.core.safe_pickle import load_tensor_pickle
from drive2sim.core.schema import Manifest
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import NoParams, Stage, StageContext
from drive2sim.stages.commands import LOG_FILE
from drive2sim.stages.prep.smpl import OUTPUT_NAME

CACHE_DIR_ENV = "D2S_SMPL_RELEASE_DIR"  # nơi giữ file zip đã tải; mặc định ~/.cache/...


@dataclass(frozen=True)
class SmplRelease:
    """Một file zip SMPL DriveStudio phát hành cho một dataset.

    Attributes:
        gdown_id: Mã Google Drive trong ``docs/datasets/human_pose.md``.
        zip_name: Tên file zip khi tải về.
        sha256: Mã băm của bản đã kiểm; file khác mã băm thì không dùng.
        member: Đường dẫn file của một cảnh trong zip, ``{index}`` là số thứ tự cảnh.
        sequences: Tên chuỗi theo đúng thứ tự đánh số của DriveStudio.
    """

    gdown_id: str
    zip_name: str
    sha256: str
    member: str
    sequences: tuple[str, ...]


RELEASES: dict[str, SmplRelease] = {
    # nuScenes v1.0-mini: DriveStudio đánh số cảnh theo thứ tự trong v1.0-mini/scene.json, như
    # ingest của tool (_nuscenes_arg). Hai cảnh 004 và 007 là dict rỗng: không có người.
    "nuscenes": SmplRelease(
        gdown_id="1Z0gJVRtPnjvusQVaW7ghZnwfycZStCZx",
        zip_name="nuscenes_processed_humanpose.zip",
        sha256="1f1977a06af02af80b4f894414e43fc939d57423d4ba8edd11ff823d1b96152b",
        member="nuscenes/processed_10Hz/mini/{index:03d}/humanpose/smpl.pkl",
        sequences=(
            "scene-0061",
            "scene-0103",
            "scene-0553",
            "scene-0655",
            "scene-0757",
            "scene-0796",
            "scene-0916",
            "scene-1077",
            "scene-1094",
            "scene-1100",
        ),
    ),
}


def release_member(manifest: Manifest) -> Optional[tuple[SmplRelease, str]]:
    """Bản phát hành và đường dẫn trong zip của cảnh; ``None`` nếu cảnh không có trong đó."""
    release = RELEASES.get(manifest.source.dataset)
    if release is None or manifest.source.sequence_id not in release.sequences:
        return None
    index = release.sequences.index(manifest.source.sequence_id)
    return release, release.member.format(index=index)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_release(release: SmplRelease) -> Path:
    """File zip của bản phát hành: dùng bản đã tải nếu đúng mã băm, không thì tải bằng gdown.

    Raises:
        RuntimeError: Nếu tải không được hoặc file tải về khác mã băm đã kiểm.
    """
    default = Path(os.environ.get("HOME", "~")).expanduser() / ".cache/drive2sim/smpl_release"
    folder = Path(os.environ.get(CACHE_DIR_ENV) or default)
    path = folder / release.zip_name
    if path.is_file() and _sha256(path) == release.sha256:
        return path
    folder.mkdir(parents=True, exist_ok=True)
    import gdown

    partial = path.with_suffix(".partial")
    if gdown.download(id=release.gdown_id, output=str(partial), quiet=True) is None:
        raise RuntimeError(f"không tải được SMPL phát hành (Google Drive {release.gdown_id})")
    if _sha256(partial) != release.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{release.zip_name} tải về khác mã băm đã kiểm: file phát hành đã đổi, cần kiểm lại"
        )
    partial.replace(path)
    return path


@register_stage("prep.smpl_release")
class PrepSmplReleaseStage(Stage):
    """Lấy SMPL phát hành sẵn của DriveStudio cho cảnh, ghi như ``prep.smpl``."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = True
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"scene": "scene"}
    outputs: ClassVar[dict[str, str]] = {"smpl": "smpl"}
    Params: ClassVar[type[BaseModel]] = NoParams

    @classmethod
    def unavailable_reason(cls, manifest: Manifest) -> Optional[str]:
        """Cảnh không có trong bản phát hành SMPL của DriveStudio thì không dùng được."""
        if release_member(manifest) is None:
            return (
                f"DriveStudio không phát hành SMPL cho cảnh {manifest.scene_id}; "
                "dùng prep.smpl để tự tính"
            )
        return None

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Tải (nếu chưa có) bản phát hành, ghi ``humanpose/smpl.pkl`` của cảnh.

        Cảnh không có người (file phát hành là dict rỗng) thì trả thư mục không có
        ``smpl.pkl``, như ``prep.smpl``; ``train.drivestudio`` khi đó tắt ``load_smpl``.

        Raises:
            RuntimeError: Nếu cảnh không có trong bản phát hành, tải lỗi hoặc file lạ.
        """
        import joblib

        found = release_member(ctx.scene.manifest)
        if found is None:
            raise RuntimeError(self.unavailable_reason(ctx.scene.manifest))
        release, member = found
        ctx.progress(0.0, "tải SMPL phát hành của DriveStudio")
        archive = fetch_release(release)
        with zipfile.ZipFile(archive) as zipped:
            smpl = load_tensor_pickle(zipped.read(member))
        if not isinstance(smpl, dict):
            raise RuntimeError(f"{member} không phải dict như smpl.pkl của DriveStudio")
        log = f"{release.zip_name} ({release.sha256[:12]}), {member}: {len(smpl)} người\n"
        (ctx.output_dir / LOG_FILE).write_text(log, "utf-8")
        if smpl:
            (ctx.output_dir / OUTPUT_NAME).mkdir(parents=True, exist_ok=True)
            joblib.dump(smpl, ctx.output_dir / OUTPUT_NAME / "smpl.pkl")
        ctx.progress(1.0, f"SMPL phát hành: {len(smpl)} người")
        return {"smpl": ctx.output_dir}
