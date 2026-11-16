"""SMPL phát hành sẵn của DriveStudio (stages/prep/smpl_release.py) và trình đọc pickle an toàn."""

import copy
import io
import os
import pickle
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
joblib = pytest.importorskip("joblib")

from drive2sim.core.safe_pickle import UnsafePickleError, load_tensor_pickle  # noqa: E402
from drive2sim.core.schema import Manifest  # noqa: E402
from drive2sim.engine.approaches import load_approach  # noqa: E402
from drive2sim.engine.registry import REGISTRY, load_builtin_stages  # noqa: E402
from drive2sim.engine.resolve import blocked_choices  # noqa: E402
from drive2sim.stages.prep import smpl_release  # noqa: E402
from drive2sim.stages.prep.smpl_release import PrepSmplReleaseStage, release_member  # noqa: E402
from tests.drive2sim.samples import MANIFEST  # noqa: E402

load_builtin_stages()


def manifest(sequence: str = "scene-0061") -> Manifest:
    data = copy.deepcopy(MANIFEST)
    data["source"]["sequence_id"] = sequence
    return Manifest.model_validate(data)


def smpl_bytes(people: int) -> bytes:
    """Pickle cùng dạng smpl.pkl của DriveStudio: dict người → dict tensor."""
    data = {
        i: {"smpl": {"betas": torch.zeros(3, 10)}, "valid_mask": torch.ones(3, dtype=torch.bool)}
        for i in range(people)
    }
    buffer = io.BytesIO()
    pickle.dump(data, buffer)
    return buffer.getvalue()


class Evil:
    """Pickle độc: lúc nạp gọi os.system."""

    def __reduce__(self) -> tuple:
        return (os.system, ("echo pwned",))


def test_safe_pickle_loads_tensors_and_rejects_code() -> None:
    loaded = load_tensor_pickle(smpl_bytes(2))
    assert sorted(loaded) == [0, 1] and loaded[0]["smpl"]["betas"].shape == (3, 10)
    with pytest.raises(UnsafePickleError, match="posix.system|os.system"):
        load_tensor_pickle(pickle.dumps(Evil()))


def test_release_member_follows_drivestudio_scene_order() -> None:
    release, member = release_member(manifest("scene-0061"))
    assert member == "nuscenes/processed_10Hz/mini/000/humanpose/smpl.pkl"
    assert release_member(manifest("scene-1100"))[1].endswith("/009/humanpose/smpl.pkl")
    assert release_member(manifest("scene-9999")) is None
    assert PrepSmplReleaseStage.unavailable_reason(manifest("scene-9999"))
    assert PrepSmplReleaseStage.unavailable_reason(manifest("scene-0061")) is None


def test_auto_picks_compute_when_scene_not_released() -> None:
    """``smpl: auto`` của OmniRe: lựa chọn bản phát hành bị khoá với cảnh không có trong đó."""
    approach = load_approach("omnire")
    blocked = blocked_choices(approach, manifest("scene-9999"), REGISTRY)
    assert [b.choice for b in blocked.get("smpl", [])] == ["prep.smpl_release"]
    assert "smpl" not in blocked_choices(approach, manifest("scene-0061"), REGISTRY)


class Ctx:
    """Ngữ cảnh giả cho stage: thư mục ra và cảnh."""

    def __init__(self, output_dir: Path, sequence: str) -> None:
        self.output_dir = output_dir
        self.scene = type("Scene", (), {"manifest": manifest(sequence)})()
        self.events: list[str] = []

    def progress(self, fraction: float, message: str = "") -> None:
        """Ghi lại tiến độ."""
        self.events.append(message)


@pytest.mark.parametrize(("sequence", "people"), [("scene-0061", 2), ("scene-0757", 0)])
def test_stage_writes_tool_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sequence: str, people: int
) -> None:
    archive = tmp_path / "release.zip"
    _, member = release_member(manifest(sequence))
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(member, smpl_bytes(people))
    monkeypatch.setattr(smpl_release, "fetch_release", lambda release: archive)
    out = tmp_path / "out"
    out.mkdir()
    assert PrepSmplReleaseStage().run(Ctx(out, sequence), {}, None) == {"smpl": out}
    written = out / "humanpose" / "smpl.pkl"
    if people:
        assert len(joblib.load(written)) == people
    else:  # cảnh không có người: như prep.smpl, không có smpl.pkl để train tắt load_smpl
        assert not written.exists()


def test_both_smpl_sources_need_the_smpl_model_file() -> None:
    """Kaggle, Colab, kiểm tra môi trường gắn file SMPL khi có stage tạo artifact smpl."""
    from drive2sim.engine.registry import produces_artifact

    assert produces_artifact(["prep.masks_segformer", "prep.smpl_release"], "smpl")
    assert produces_artifact(["prep.smpl"], "smpl")
    assert not produces_artifact(["prep.masks_segformer", "train.drivestudio"], "smpl")
