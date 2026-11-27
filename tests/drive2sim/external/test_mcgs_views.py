"""Kiểm tra ``external/mcgs/views.py`` và stage ``eval.mcgs`` bằng một cảnh giả nhỏ.

- ``fit_alignment`` tìm lại đúng phép đồng dạng giữa hệ SLAM và world; ``to_map_viewmat`` đặt
  camera thật vào hệ bản đồ co 0,2 lần.
- ``fisheye_theta`` giải đúng góc lệch trục của mô hình fisheye OpenCV.
- ``build_views`` ghi ``views.npz`` có mask thân xe và mask góc fisheye.
- ``EvalMcgsStage.run`` nối các bước với bước vẽ giả (không cần GPU).
"""

import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from drive2sim.core.schema import Manifest, Split  # noqa: E402
from drive2sim.engine.stage import Scene  # noqa: E402
from drive2sim.external import mcgs  # noqa: E402
from drive2sim.external.mcgs import views  # noqa: E402
from drive2sim.stages.eval import mcgs as eval_mcgs  # noqa: E402
from tests.drive2sim.samples import MANIFEST, SPLIT  # noqa: E402

FRAMES = 10
SIZE = (24, 32)  # cao, rộng của ảnh giả
FISHEYE = [20.0, 20.0, 16.0, 12.0, 0.01, -0.002, 0.0, 0.0]
PINHOLE = [30.0, 30.0, 16.0, 12.0, 0, 0, 0, 0, 0]


def rotation_z(angle: float) -> "np.ndarray":
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def save(path: Path, matrix: "np.ndarray") -> None:
    np.savetxt(path, matrix)


def make_scene(root: Path) -> None:
    """Cảnh có 2 camera (pinhole CAM_FRONT, fisheye CAM_FISH), xe đi cong trong 10 khung."""
    for name in ("ego_pose", "extrinsics", "intrinsics", "images"):
        (root / name).mkdir(parents=True)
    for frame in range(FRAMES):
        ego = np.eye(4)
        ego[:3, :3] = rotation_z(0.05 * frame)
        ego[:3, 3] = [3.0 * frame, 0.4 * frame * frame, 0.1 * frame]
        save(root / "ego_pose" / f"{frame:03d}.txt", ego)
        for index in range(2):
            rng = np.random.default_rng(frame * 2 + index)
            image = rng.integers(0, 255, size=(*SIZE, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "images" / f"{frame:03d}_{index}.png"), image)
    for index in range(2):
        cam_to_ego = np.eye(4)
        cam_to_ego[:3, 3] = [0.5 * index, 0.0, 1.5]
        save(root / "extrinsics" / f"{index}.txt", cam_to_ego)
    np.savetxt(root / "intrinsics" / "0.txt", PINHOLE)
    np.savetxt(root / "intrinsics" / "1.txt", FISHEYE)


def poses_of(scene: Path) -> "views.ScenePoses":
    return views.ScenePoses(scene, "kitti360", FRAMES)


SIM = {"scale": 4.0, "R": rotation_z(0.9), "t": np.array([5.0, -3.0, 2.0])}


def write_trajectory(root: Path, path: Path, frames: list[int]) -> None:
    """TUM của camera 0 trong hệ SLAM: ``slam = Rᵀ (world − t) / scale`` (quaternion đơn vị)."""
    lines = []
    for frame in frames:
        world = views.camera_to_world(root, frame, 0)[:3, 3]
        slam = SIM["R"].T @ (world - SIM["t"]) / SIM["scale"]
        lines.append(f"{float(frame)} {slam[0]} {slam[1]} {slam[2]} 0 0 0 1")
    path.write_text("\n".join(lines) + "\n")


def test_alignment_recovers_the_similarity(tmp_path: Path) -> None:
    make_scene(tmp_path / "scene")
    write_trajectory(tmp_path / "scene", tmp_path / "traj.txt", [0, 2, 3, 5, 7, 9])
    alignment = views.fit_alignment(tmp_path / "traj.txt", poses_of(tmp_path / "scene"), 0)
    assert alignment["scale"] == pytest.approx(SIM["scale"])
    assert alignment["R"] == pytest.approx(SIM["R"], abs=1e-9)
    assert alignment["t"] == pytest.approx(SIM["t"], abs=1e-9)
    assert alignment["ate_rmse_m"] == pytest.approx(0.0, abs=1e-9)
    assert alignment["keyframes"] == 6


def test_alignment_needs_matching_keyframes(tmp_path: Path) -> None:
    make_scene(tmp_path / "scene")
    write_trajectory(tmp_path / "scene", tmp_path / "traj.txt", [0, 4])
    with pytest.raises(ValueError, match="khớp với khung"):
        views.fit_alignment(tmp_path / "traj.txt", poses_of(tmp_path / "scene"), 0)
    # timestamp là giây, 10 khung mỗi giây: frames_per_stamp = 10
    write_trajectory(tmp_path / "scene", tmp_path / "traj2.txt", [0, 1, 2, 3, 4])
    text = (tmp_path / "traj2.txt").read_text().splitlines()
    scaled = [f"{float(line.split()[0]) / 10} " + " ".join(line.split()[1:]) for line in text]
    (tmp_path / "traj2.txt").write_text("\n".join(scaled) + "\n")
    found = views.fit_alignment(
        tmp_path / "traj2.txt", poses_of(tmp_path / "scene"), 0, frames_per_stamp=10.0
    )
    assert found["keyframes"] == 5 and found["scale"] == pytest.approx(SIM["scale"])


def test_map_viewmat_puts_the_camera_in_the_scaled_slam_frame(tmp_path: Path) -> None:
    make_scene(tmp_path / "scene")
    write_trajectory(tmp_path / "scene", tmp_path / "traj.txt", list(range(FRAMES)))
    alignment = views.fit_alignment(tmp_path / "traj.txt", poses_of(tmp_path / "scene"), 0)
    c2w_world = views.camera_to_world(tmp_path / "scene", 6, 1)  # camera thứ hai (lệch 0,5 m)
    viewmat = views.to_map_viewmat(c2w_world, alignment)
    c2w_map = np.linalg.inv(viewmat.astype(np.float64))
    slam_position = SIM["R"].T @ (c2w_world[:3, 3] - SIM["t"]) / SIM["scale"]
    assert c2w_map[:3, 3] == pytest.approx(mcgs.POSE_SCALE * slam_position, abs=1e-5)
    assert c2w_map[:3, :3] == pytest.approx(SIM["R"].T @ c2w_world[:3, :3], abs=1e-6)
    # camera 0 trùng quỹ đạo SLAM: vị trí trong hệ bản đồ là 0,2 lần pose đã ghi trong TUM
    c2w_ref = views.camera_to_world(tmp_path / "scene", 3, 0)
    ref_map = np.linalg.inv(views.to_map_viewmat(c2w_ref, alignment).astype(np.float64))
    traj_row = np.array(
        [float(v) for v in (tmp_path / "traj.txt").read_text().splitlines()[3].split()]
    )
    assert ref_map[:3, 3] == pytest.approx(mcgs.POSE_SCALE * traj_row[1:4], abs=1e-5)


def test_fisheye_theta_inverts_the_opencv_model() -> None:
    intrinsics = np.array([[20.0, 0, 16.0], [0, 20.0, 12.0], [0, 0, 1.0]])
    radial = np.array(FISHEYE[4:8])
    theta = views.fisheye_theta(intrinsics, radial, *SIZE)
    assert theta[12, 16] == pytest.approx(0.0, abs=1e-9)  # tâm ảnh
    for theta_true in (0.3, 0.9, 1.4):
        t2 = theta_true**2
        theta_d = theta_true * (1 + radial[0] * t2 + radial[1] * t2**2)
        column = int(round(16 + 20.0 * theta_d))
        if column < SIZE[1]:
            assert theta[12, column] == pytest.approx(theta_true, abs=0.05)  # lấy mẫu theo pixel
    assert theta.shape == SIZE


def test_read_intrinsics_checks_counts(tmp_path: Path) -> None:
    make_scene(tmp_path)
    matrix, radial = views.read_intrinsics(tmp_path, 1, "opencv_fisheye")
    assert matrix[0, 0] == 20.0 and radial.tolist() == FISHEYE[4:8]
    _, zero = views.read_intrinsics(tmp_path, 0, "opencv_pinhole")
    assert zero.tolist() == [0.0] * 4
    with pytest.raises(ValueError, match="cần 8 số"):
        views.read_intrinsics(tmp_path, 0, "opencv_fisheye")
    with pytest.raises(ValueError, match="chưa hỗ trợ"):
        views.read_intrinsics(tmp_path, 0, "ftheta")


def test_build_views_masks(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    make_scene(scene)
    write_trajectory(scene, tmp_path / "traj.txt", list(range(FRAMES)))
    alignment = views.fit_alignment(tmp_path / "traj.txt", poses_of(scene), 0)
    masks = tmp_path / "masks"
    masks.mkdir()
    ego = np.zeros(SIZE, dtype=np.uint8)
    ego[:4, :] = 255
    cv2.imwrite(str(masks / "CAM_FISH.png"), ego)
    info = views.build_views(
        scene,
        poses_of(scene),
        ["CAM_FRONT", "CAM_FISH"],
        {"CAM_FRONT": "opencv_pinhole", "CAM_FISH": "opencv_fisheye"},
        [(8, "CAM_FRONT"), (8, "CAM_FISH")],
        alignment,
        tmp_path / "views.npz",
        max_angle=0.5,
        ego_masks_dir=masks,
    )
    assert [v["camera_id"] for v in info] == ["CAM_FRONT", "CAM_FISH"]
    assert info[1]["position"] == pytest.approx(views.camera_to_world(scene, 8, 1)[:3, 3].tolist())
    data = np.load(tmp_path / "views.npz", allow_pickle=True)
    assert data["camera"].tolist() == ["CAM_FRONT", "CAM_FISH"]
    assert data["model"].tolist() == ["pinhole", "fisheye"]
    assert data["viewmat"].shape == (2, 4, 4) and data["radial"].shape == (2, 4)
    pinhole_mask, fisheye_mask = data["mask"]
    assert not pinhole_mask.any()  # pinhole không có mask thân xe hay góc
    assert fisheye_mask[:4].all()  # thân xe
    theta = views.fisheye_theta(data["K"][1], data["radial"][1], *SIZE)
    assert (fisheye_mask[4:] == (theta[4:] > 0.5)).all()  # góc quá max_angle
    assert data["image"][0].shape == (*SIZE, 3) and data["image"][0].dtype == np.uint8
    with pytest.raises(FileNotFoundError):
        views.build_views(
            scene, poses_of(scene), ["CAM_FRONT", "CAM_FISH"],
            {"CAM_FRONT": "opencv_pinhole", "CAM_FISH": "opencv_fisheye"},
            [(99, "CAM_FRONT")], alignment, tmp_path / "v2.npz",
        )  # fmt: skip


class Context:
    """Ngữ cảnh tối thiểu cho ``Stage.run`` của ``eval.mcgs``."""

    def __init__(self, output_dir: Path, scene_root: Path) -> None:
        manifest = Manifest.model_validate(
            dict(
                MANIFEST,
                scene_id="kitti360_0003",
                source=dict(MANIFEST["source"], dataset="kitti360"),
            )
        )
        self.output_dir = output_dir
        self.scene = Scene("kitti360_0003", scene_root, manifest)
        split = dict(SPLIT, scene_id="kitti360_0003", cameras=["CAM_FRONT", "CAM_FISH"])
        split["test"] = [[8, "CAM_FRONT"], [8, "CAM_FISH"]]
        self.split = Split.model_validate(split)
        self.events: list[str] = []

    def progress(self, fraction: float, message: str = "") -> None:
        """Ghi lại thông điệp tiến độ."""
        self.events.append(message)

    def should_stop(self) -> bool:
        """Không bao giờ huỷ."""
        return False


def test_eval_stage_runs_with_a_fake_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    make_scene(scene)
    model = tmp_path / "mcgs_out"
    model.mkdir()
    write_trajectory(scene, model / mcgs.OUTPUT_TRAJECTORY, list(range(FRAMES)))
    (model / mcgs.OUTPUT_GAUSSIANS).write_bytes(b"ply")
    out = tmp_path / "out"
    out.mkdir()
    commands: list[list[str]] = []

    def fake_render(command: list[str], log: Path, **kwargs: object) -> None:
        commands.append(command)
        assert Path(command[-4]).name == mcgs.OUTPUT_GAUSSIANS
        assert Path(command[-3]).is_file()  # views.npz có mặt lúc vẽ
        metrics = {
            "CAM_FRONT": {"psnr_mean": 24.0, "ssim_mean": 0.8, "lpips_mean": 0.2, "views": 1},
            "CAM_FISH": {"psnr_mean": 18.0, "ssim_mean": 0.6, "lpips_mean": 0.4, "views": 1},
            "_summary": {
                "gaussians": 1000, "psnr_mean": 21.0, "ssim_mean": 0.7, "lpips_mean": 0.3,
                "render_resolution": "32x24", "render_fps": 10.0, "peak_vram_gib": 1.0,
            },
        }  # fmt: skip
        (out / "metrics.json").write_text(json.dumps(metrics))
        views_result = [
            {"camera": "CAM_FRONT", "psnr": 24.0, "ssim": 0.8, "lpips": 0.2, "coverage": 0.8},
            {"camera": "CAM_FISH", "psnr": 18.0, "ssim": 0.6, "lpips": 0.4, "coverage": 0.4},
        ]
        (out / "per_view.json").write_text(json.dumps(views_result))

    monkeypatch.setattr(eval_mcgs, "run_logged", fake_render)
    stage = eval_mcgs.EvalMcgsStage()
    params = eval_mcgs.EvalMcgsParams(
        model_dir=str(model), held_out=True, reference_camera="CAM_FRONT"
    )
    assert stage.run(Context(out, scene), {}, params) == {"report": out}
    from drive2sim.core import files
    from drive2sim.core.schema import Summary

    summary = files.load(out / "summary.json", Summary)
    assert summary.overall.psnr == 21.0 and set(summary.per_camera) == {"CAM_FRONT", "CAM_FISH"}
    assert summary.protocol.version == "ncore-0.3" and summary.num_gaussians == 1000
    assert summary.coverage == pytest.approx(0.6)
    rows = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert [(r["frame_index"], r["camera_index"]) for r in rows] == [(8, 0), (8, 1)]
    assert rows[1]["psnr"] == 18.0 and len(rows[0]["position"]) == 3 and rows[0]["depth"] is None
    assert [r["coverage"] for r in rows] == [0.8, 0.4]
    note = json.loads((out / "mcgs_eval.json").read_text())
    assert note["held_out"] and note["keyframes"] == FRAMES
    assert note["alignment_scale"] == pytest.approx(SIM["scale"])
    assert not (out / "views.npz").exists()  # file lớn bị xoá sau khi vẽ
    with pytest.raises(ValueError, match="reference_camera"):
        stage.run(Context(out, scene), {}, eval_mcgs.EvalMcgsParams(model_dir=str(model)))
    with pytest.raises(ValueError, match="model_dir"):
        stage.run(Context(out, scene), {}, eval_mcgs.EvalMcgsParams())
    with pytest.raises(FileNotFoundError, match="demo.py"):
        stage.run(Context(out, scene), {}, eval_mcgs.EvalMcgsParams(model_dir=str(tmp_path)))


def test_scene_poses_dispatch_by_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """KITTI-360 đọc trực tiếp; dataset khác gọi ``get_camera2worlds`` của loader (nhớ lại)."""
    make_scene(tmp_path)
    direct = views.ScenePoses(tmp_path, "kitti360", FRAMES)
    assert direct.c2w(4, 1) == pytest.approx(views.camera_to_world(tmp_path, 4, 1))
    with pytest.raises(ValueError, match="physicalai"):
        views.ScenePoses(tmp_path, "physicalai", FRAMES)
    calls: list[tuple] = []

    class FakeCamera:
        @classmethod
        def get_camera2worlds(
            cls, data_path: str, cam_id: int, start: int, end: int
        ) -> "np.ndarray":
            calls.append((data_path, cam_id, start, end))
            return np.stack([np.eye(4) * (frame + 1) for frame in range(start, end)])

    class FakeModule:
        NuScenesCameraData = FakeCamera

    requested: list[str] = []

    def fake_import(name: str) -> object:
        requested.append(name)
        return FakeModule

    monkeypatch.setattr(views.importlib, "import_module", fake_import)
    loader = views.ScenePoses(tmp_path, "nuscenes", FRAMES)
    assert loader.c2w(3, 2)[0, 0] == 4.0 and loader.c2w(5, 2)[0, 0] == 6.0
    assert calls == [(str(tmp_path), 2, 0, FRAMES)]  # một lần gọi cho mỗi camera, từ khung 0
    assert requested == ["drive2sim.recon.data.loaders.nuscenes"]
    assert set(views.SUPPORTED_DATASETS) == {
        "kitti", "kitti360", "nuscenes", "waymo", "argoverse2", "pandaset", "nuplan",
    }  # fmt: skip


def test_build_views_finds_images_with_any_extension(tmp_path: Path) -> None:
    make_scene(tmp_path / "scene")
    scene = tmp_path / "scene"
    image = cv2.imread(str(scene / "images" / "008_0.png"))
    (scene / "images" / "008_0.png").unlink()
    cv2.imwrite(str(scene / "images" / "008_0.jpg"), image)  # nuScenes, Waymo... ghi jpg
    write_trajectory(scene, tmp_path / "traj.txt", list(range(FRAMES)))
    alignment = views.fit_alignment(tmp_path / "traj.txt", poses_of(scene), 0)
    info = views.build_views(
        scene, poses_of(scene), ["CAM_FRONT", "CAM_FISH"],
        {"CAM_FRONT": "opencv_pinhole", "CAM_FISH": "opencv_fisheye"},
        [(8, "CAM_FRONT")], alignment, tmp_path / "v.npz",
    )  # fmt: skip
    assert info[0]["frame"] == 8


def test_pinhole_distortion_rectifies_rgb_and_official_mask_with_same_map(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    make_scene(scene)
    coefficients = np.array([0.7, 0.03, 0.02, -0.01, 0.0])
    save(scene / "intrinsics" / "0.txt", np.array(PINHOLE[:4] + coefficients.tolist()))
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (*SIZE, 3), np.uint8)
    cv2.imwrite(str(scene / "images" / "008_0.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    (scene / "ego_masks").mkdir()
    mask = np.zeros(SIZE, np.uint8)
    mask[10:, 5:20] = 255
    cv2.imwrite(str(scene / "ego_masks" / "CAM_FRONT.png"), mask)
    alignment = {"scale": 1.0, "R": np.eye(3), "t": np.zeros(3)}
    output = tmp_path / "views.npz"
    views.build_views(
        scene,
        poses_of(scene),
        ["CAM_FRONT", "CAM_FISH"],
        {"CAM_FRONT": "opencv_pinhole"},
        [(8, "CAM_FRONT")],
        alignment,
        output,
        dataset="custom",
    )
    intrinsics, _ = views.read_intrinsics(scene, 0, "opencv_pinhole")
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsics, coefficients, None, intrinsics, (SIZE[1], SIZE[0]), cv2.CV_32FC1
    )
    expected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    allowed = cv2.remap((mask == 0).astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST)
    with np.load(output, allow_pickle=True) as result:
        np.testing.assert_array_equal(result["image"][0], expected)
        np.testing.assert_array_equal(result["mask"][0], allowed == 0)
    assert not np.array_equal(image, expected)
