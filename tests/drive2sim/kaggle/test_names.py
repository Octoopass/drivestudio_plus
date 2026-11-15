"""Kiểm tra quy tắc đặt tên notebook và dataset trên Kaggle (``drive2sim/kaggle/names.py``).

- tên đọc là hiểu: cảnh, approach, renderer, thuật toán phụ trợ, run, lần đẩy;
- renderer luôn có mặt, kể cả renderer mặc định; tên không quá 50 ký tự, không cắt nửa chữ;
- run đẩy theo tên cũ (băm) vẫn chạy tiếp được: lần đẩy sau gắn đúng notebook cũ.
"""

import json
from pathlib import Path

import pytest

from drive2sim.kaggle import executor, names
from tests.drive2sim.kaggle.test_kaggle import RUN_ID, FakeKaggle, make_workdir

DEFAULTS = {"renderer": "3dgs", "campose": False, "fastgs": False, "fastergs": False}


def test_scene_label() -> None:
    assert names.scene_label("nuscenes_scene-0061") == "nuscenes-0061"
    assert names.scene_label("kitti360_2013_05_28_drive_0003_sync") == "kitti360-0003"
    assert names.scene_label("waymo_10017090168044687777") == "waymo-10017090168044687777"
    assert names.scene_label("demo_street") == "demo-street"


def test_train_kernel_always_names_renderer() -> None:
    run = "20261114-123210-omnire-f8a6"
    assert (
        names.train_kernel("nuscenes_scene-0061", run, 1, DEFAULTS, DEFAULTS)
        == "d2s-nuscenes-0061-omnire-3dgs-f8a6-1"
    )
    geer = {**DEFAULTS, "renderer": "3dgeer", "fastergs": True}
    assert (
        names.train_kernel("nuscenes_scene-0061", run, 2, geer, DEFAULTS)
        == "d2s-nuscenes-0061-omnire-3dgeer-fastergs-f8a6-2"
    )
    both = {**DEFAULTS, "fastgs": True, "fastergs": True}
    # Quá 50 ký tự với tiền tố d2s-: bỏ tiền tố, giữ đủ biến thể.
    assert (
        names.train_kernel("nuscenes_scene-0061", run, 2, both, DEFAULTS)
        == "nuscenes-0061-omnire-3dgs-fastgs-fastergs-f8a6-2"
    )
    long = {**DEFAULTS, "renderer": "3dgeer", "campose": True, "fastgs": True, "fastergs": True}
    name = names.train_kernel("kitti360_2013_05_28_drive_0003_sync", run, 12, long, DEFAULTS)
    assert len(name) <= names.MAX_LENGTH and name.endswith("-f8a6-12")
    assert all(
        word in ("3dgeer", "campose", "fastgs", "fastergs") for word in name.split("-")[3:-2]
    )


def test_variant_label_marks_disabled_default() -> None:
    assert names.variant_label({"lidar": False}, {"lidar": True}) == "no-lidar"
    assert names.variant_label({}, {}) == "base"


def test_scene_dataset_and_ingest_kernel() -> None:
    assert (
        names.scene_dataset("nuscenes_scene-0061", "0.1.0+96876ad")
        == "d2s-scene-nuscenes-0061-96876ad"
    )
    assert (
        names.ingest_kernel(None, "scene-0061", "alice/d2s-nuscenes-mini-raw")
        == "d2s-ingest-nuscenes-mini-0061"
    )
    assert (
        names.ingest_kernel(None, None, "alice/d2s-raw-kitti360-0003")
        == "d2s-ingest-kitti360-0003"
    )
    assert names.ingest_kernel("kitti360_0003_a", None, "x/y") == "d2s-ingest-kitti360-0003-a"


def test_resume_from_legacy_hash_named_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        executor,
        "code_source",
        lambda branch="drive2sim": ("c0ffee" * 6 + "abcd", "https://x/r.git"),
    )
    monkeypatch.setenv("D2S_KAGGLE_SMPL_DATASET", "alice/smpl-model")
    workdir = make_workdir(tmp_path / "work")
    fake = FakeKaggle(tmp_path / "kaggle")
    # kaggle.json ghi từ trước khi có trường kernels: hai lần đẩy theo tên băm.
    legacy = f"alice/d2s-{executor._hash8(RUN_ID)}"
    state = {
        "username": "alice",
        "kernel": f"{legacy}-2",
        "attempt": 2,
        "dataset": "alice/d2s-1081cb84",
        "commit": "c0ffee",
        "pushed_utc": "2026-11-14T00:00:00+00:00",
        "pulled_utc": None,
    }
    (workdir.find_run(RUN_ID) / "d2s" / executor.STATE_FILE).write_text(json.dumps(state))
    third = executor.push_run(workdir, RUN_ID, client=fake)
    # run mẫu không ghi lựa chọn nào (options rỗng), nên biến thể là base
    assert third.kernel == "alice/d2s-nuscenes-0061-omnire-base-ab12-3"
    assert fake.kernels[third.kernel]["metadata"]["kernel_sources"] == [
        f"{legacy}-2",
        f"{legacy}-1",
    ]
    assert third.kernels == [f"{legacy}-1", f"{legacy}-2", third.kernel]
