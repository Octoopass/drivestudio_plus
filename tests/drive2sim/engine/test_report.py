"""Kiểm tra cách ``drive2sim/engine/report.py`` tính thời gian dựng cảnh và đọc file PLY.

- ``recon_time``: cộng thời gian từ stage đầu tới hết ``export.ply_world``; stage dùng lại cache
  tính thời gian của lần chạy gốc; thiếu số đó hoặc chưa xuất PLY thì trả ``None``.
- ``ply_stats``: đọc số Gaussian ở phần đầu file PLY và dung lượng file.
"""

from pathlib import Path
from typing import Optional

from drive2sim.core.schema import Manifest, RunApproach, RunRecord, Split, StageRecord
from drive2sim.engine.report import (
    ReportRow,
    avg_error,
    ply_stats,
    preview_stats,
    recon_time,
    rows_to_csv,
    train_params,
)


def make_record(stages: list[StageRecord]) -> RunRecord:
    return RunRecord(
        schema_version="d2s-run/0.3",
        run_id="20261112-081000-omnire-a1b2",
        scene_id="nuscenes_scene-0061",
        split_id="every8_allcams",
        approach=RunApproach(approach_id="omnire", version="0.1.0"),
        level="L1",
        executor="local",
        status="succeeded",
        stages=stages,
    )


def stage(stage_id: str, duration: float, **kwargs: object) -> StageRecord:
    return StageRecord(stage_id=stage_id, status="succeeded", duration_s=duration, **kwargs)


def test_recon_time_counts_original_time_of_cached_stages() -> None:
    record = make_record(
        [
            stage("prep.masks_segformer", 0, cache_hit=True, reused_duration_s=810.0),
            stage("prep.smpl", 1920.0),
            stage("train.drivestudio", 4380.0),
            stage("eval.standard", 3.0),
            stage("export.ply_world", 41.0),
            stage("export.preview", 30.0),  # sau khi đã có PLY: không tính
        ]
    )
    assert recon_time(record) == 810 + 1920 + 4380 + 3 + 41


def test_recon_time_unknown() -> None:
    no_ply = make_record([stage("train.drivestudio", 10.0)])
    assert recon_time(no_ply) is None
    old_cache = make_record(
        [stage("prep.smpl", 0, cache_hit=True), stage("export.ply_world", 1.0)]
    )
    assert recon_time(old_cache) is None


def test_ply_stats(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1234\n"
        b"property float x\nend_header\n"
    )
    ply.write_bytes(header + b"\0" * (2 * 1024 * 1024))
    count, size_mb = ply_stats(ply)
    assert count == 1234 and size_mb == 2.0
    assert ply_stats(tmp_path / "missing.ply") == (None, None)


def test_preview_stats_and_csv(tmp_path: Path) -> None:
    folder = tmp_path / "export.preview"
    (folder / "actors").mkdir(parents=True)
    for path, count in (
        (folder / "background.ply", 1000),
        (folder / "actors" / "rigid_0000.ply", 50),
    ):
        path.write_bytes(f"ply\nelement vertex {count}\nend_header\n".encode())
    assert preview_stats(folder) == (None, None)  # chưa có preview.json: chưa xuất gói
    (folder / "preview.json").write_text("{}")
    count, size_mb = preview_stats(folder)
    assert count == 1050 and size_mb is not None
    times = {"train.drivestudio": {"seconds": 1.0, "reused": False}}
    row = ReportRow(run_id="r", scene_id="s", split_id="x", approach_id="a", stage_times=times)
    header = rows_to_csv([row]).splitlines()[0]
    assert "stage_times" not in header and "preview_gaussians" in header


def test_preview_stats_counts_spz(tmp_path: Path) -> None:
    import gzip
    import struct

    folder = tmp_path / "export.preview"
    (folder / "humans").mkdir(parents=True)
    for path, count in (
        (folder / "background.spz", 3000),
        (folder / "humans" / "smpl_0000.spz", 70),
    ):
        path.write_bytes(
            gzip.compress(struct.pack("<IIIBBBB", 0x5053474E, 3, count, 3, 12, 0, 0) + b"\0" * 8)
        )
    (folder / "humans" / "broken.spz").write_bytes(b"not gzip")  # file hỏng không tính, không lỗi
    (folder / "preview.json").write_text("{}")
    count, _ = preview_stats(folder)
    assert count == 3070


def test_avg_error_matches_mip_nerf() -> None:
    # Mip-NeRF trên Blender: PSNR 33.09, SSIM 0.961, LPIPS 0.043, Avg báo cáo 0.016
    assert round(avg_error(33.09, 0.961, 0.043), 3) == 0.016
    # tốt hơn ở cả ba metric thì Avg thấp hơn
    assert avg_error(28.3, 0.874, 0.183) < avg_error(24.4, 0.803, 0.271)


def test_split_info_reads_split_and_manifest(tmp_path: Path) -> None:
    """Cỡ ảnh gốc theo camera của split trong manifest, số ảnh train/test; thiếu cảnh thì rỗng."""
    from drive2sim.core import files
    from drive2sim.core.schema import Manifest, Split
    from drive2sim.engine.report import split_info
    from drive2sim.engine.workdir import Workdir
    from tests.drive2sim.samples import MANIFEST, SPLIT

    workdir = Workdir(tmp_path)
    scene = MANIFEST["scene_id"]
    assert split_info(workdir, scene, SPLIT["split_id"]) == {}
    files.dump(Manifest.model_validate(MANIFEST), workdir.scene_dir(scene) / "manifest.yaml")
    files.dump(Split.model_validate(SPLIT), workdir.split_path(scene, SPLIT["split_id"]))
    info = split_info(workdir, scene, SPLIT["split_id"])
    assert info["source_resolution"] == "1600x900"
    assert (info["train_images"], info["test_images"]) == (len(SPLIT["train"]), len(SPLIT["test"]))
    assert info["test_every"] == 8  # every_nth_frame:8


def test_train_params_shows_only_non_default_params_not_in_approach_options() -> None:
    """Hai run cùng cấu hình approach phân biệt được bằng tham số stage khác mặc định."""
    from pydantic import BaseModel

    from drive2sim.engine.registry import StageRegistry

    class Params(BaseModel):
        method: str  # bắt buộc: không có giá trị mặc định
        renderer: str = "3dgs"
        num_iters: Optional[int] = None
        refine_interval: Optional[int] = None
        densify_stats: bool = False
        overrides: dict = {}

    class Train:
        pass

    Train.Params = Params  # type: ignore[attr-defined]
    registry = StageRegistry()
    registry._stages = {"train.drivestudio": Train}  # type: ignore[attr-defined]
    record = make_record(
        [
            StageRecord(
                stage_id="train.drivestudio",
                status="succeeded",
                params={
                    "method": "omnire",
                    "renderer": "3dgeer",  # lựa chọn của approach: đã hiện ở cấu hình
                    "num_iters": 3000,  # có cột riêng
                    "refine_interval": 50,
                    "densify_stats": True,  # chỉ đo đạc
                    "overrides": {"model.Background.ctrl.densify_grad_thresh": 0.0001},
                },
            )
        ]
    )
    record.approach.options = {"renderer": "3dgeer"}
    assert train_params(record, registry) == {
        "refine_interval": 50,
        "model.Background.ctrl.densify_grad_thresh": 0.0001,
    }


def test_report_gpu_is_the_training_machine(tmp_path: Path) -> None:
    """Train trên Kaggle (T4), xuất lại gói xem trước trên máy cá nhân (T2000): GPU là T4."""
    from drive2sim.core import files
    from drive2sim.core.schema import SCHEMA_VERSIONS, EnvRecord, Machine
    from drive2sim.engine.report import report_rows
    from drive2sim.engine.workdir import Workdir
    from tests.drive2sim.samples import MANIFEST, SPLIT

    workdir = Workdir(tmp_path)
    scene_id, split_id = MANIFEST["scene_id"], SPLIT["split_id"]
    manifest_path = workdir.scene_dir(scene_id) / "manifest.yaml"
    files.dump(Manifest.model_validate(MANIFEST), manifest_path)
    files.dump(Split.model_validate(SPLIT), workdir.split_path(scene_id, split_id))
    stages = [
        stage("train.drivestudio", 100.0),
        stage("export.ply_world", 3.0),
        stage("export.preview", 4.0),
    ]
    record = make_record(stages).model_copy(update={"split_id": split_id})
    run_dir = workdir.run_dir(record.scene_id, "omnire", record.run_id)
    files.dump(record, run_dir / "d2s" / "run.json")
    env = EnvRecord(
        schema_version=SCHEMA_VERSIONS["env"],
        code_commit="abc",
        python="3.10",
        executor="kaggle",
        gpu="Quadro T2000",
        machines=[
            Machine(gpu="Tesla T4", stages=["train.drivestudio", "export.ply_world"]),
            Machine(gpu="Quadro T2000", stages=["export.preview"]),
        ],
    )
    files.dump(env, run_dir / "d2s" / "env.json")
    assert [row.gpu for row in report_rows(workdir)] == ["Tesla T4"]
    # env.json cũ chưa ghi stage theo máy: gộp mọi máy như trước
    env.machines = [Machine(gpu="Tesla T4"), Machine(gpu="Quadro T2000")]
    files.dump(env, run_dir / "d2s" / "env.json")
    assert [row.gpu for row in report_rows(workdir)] == ["Tesla T4 + Quadro T2000"]


def test_retired_approach_results_remain_readable(tmp_path: Path) -> None:
    from drive2sim.core import files
    from drive2sim.engine.report import report_rows
    from drive2sim.engine.run import list_runs
    from drive2sim.engine.workdir import Workdir
    from tests.drive2sim.samples import MANIFEST, SPLIT

    workdir = Workdir(tmp_path)
    from drive2sim.core.schema import Manifest, Split

    files.dump(
        Manifest.model_validate(MANIFEST),
        workdir.scene_dir(MANIFEST["scene_id"]) / "manifest.yaml",
    )
    files.dump(
        Split.model_validate(SPLIT), workdir.split_path(MANIFEST["scene_id"], SPLIT["split_id"])
    )
    record = make_record([StageRecord(stage_id="train.gsplat_3dgut", status="succeeded")])
    record.run_id = record.run_id.replace("-omnire-", "-gsplat_3dgut-")
    record.approach.approach_id = "gsplat_3dgut"
    folder = workdir.run_dir(record.scene_id, record.approach.approach_id, record.run_id)
    files.dump(record, folder / "d2s" / "run.json")
    assert [item.run_id for item in list_runs(workdir)] == [record.run_id]
    rows = report_rows(workdir)
    assert len(rows) == 1 and rows[0].approach_id == "gsplat_3dgut"
    assert rows[0].train_params == {}
