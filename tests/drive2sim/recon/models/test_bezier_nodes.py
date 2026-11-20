"""Kiểm tra ``BezierNodes`` (node Gaussian vật động theo đường Bézier của BezierGS) trên CPU.

Dựng node từ vật giả (box đi thẳng và quay quanh trục z) rồi kiểm: vị trí Gaussian bám theo box ở
các khung train, bỏ vật quá ít khung hoặc quá ít điểm, ẩn ngoài khoảng xuất hiện nhưng vẫn hiện ở
khung bị che giữa chừng, vận tốc khớp sai phân, densify giữ nhất quán shape, tên nhóm tham số,
khôi phục từ state dict khác kích thước và ICC loss.
"""

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch3d")

from omegaconf import OmegaConf  # noqa: E402

from drive2sim.core.config import load_config  # noqa: E402
from drive2sim.recon.models.nodes import BezierNodes  # noqa: E402
from drive2sim.recon.models.nodes import bezier_math as bm  # noqa: E402

NUM_FRAMES = 10
TIMESTAMPS = torch.linspace(0, 1, NUM_FRAMES)


def make_bezier(frame_infos: list, num_pts: list = None) -> "BezierNodes":
    """Dựng node từ một danh sách ``frame_info`` (mỗi phần tử là một vật) và số điểm mỗi vật."""
    cfg = OmegaConf.create(load_config("drive2sim/configs/methods/streetgs.yaml"))
    model_cfg = cfg.model.RigidNodes
    model_cfg.ctrl = OmegaConf.merge(cfg.trainer.gaussian_ctrl_general_cfg, model_cfg.ctrl)
    model_cfg.optim = OmegaConf.merge(cfg.trainer.gaussian_optim_general_cfg, model_cfg.optim)
    model_cfg.bezier = {
        "order": 3,
        "min_instance_pts": 30,
        "opacity_init": 0.01,
        "fit_rounds": 100,
    }
    model = BezierNodes(
        **model_cfg,
        class_name="BezierNodes",
        scene_scale=30.0,
        scene_origin=torch.zeros(3),
        num_train_images=12,
        device=torch.device("cpu"),
    )
    model.register_normalized_timestamps(TIMESTAMPS)
    num_pts = num_pts or [40] * len(frame_infos)
    torch.manual_seed(0)
    instances = {}
    for i, (info, count) in enumerate(zip(frame_infos, num_pts)):
        frames = torch.arange(NUM_FRAMES, dtype=torch.float32)
        angle = 0.05 * frames
        poses = torch.eye(4).repeat(NUM_FRAMES, 1, 1)
        poses[:, 0, 0], poses[:, 0, 1] = torch.cos(angle), -torch.sin(angle)
        poses[:, 1, 0], poses[:, 1, 1] = torch.sin(angle), torch.cos(angle)
        poses[:, 0, 3] = 2.0 * frames + 20.0 * i
        instances[i] = {
            "class_name": "Car",
            "pts": torch.randn(count, 3),
            "colors": torch.rand(count, 3),
            "poses": poses,
            "size": torch.tensor([4.0, 2.0, 1.5]),
            "frame_info": torch.tensor(info, dtype=torch.bool),
            "num_pts": count,
        }
    model.create_from_pcd(instances)
    model.instances_for_test = instances
    return model


def all_frames() -> list:
    return [True] * NUM_FRAMES


def fake_camera() -> types.SimpleNamespace:
    return types.SimpleNamespace(camtoworlds=torch.eye(4))


def test_means_follow_box_at_training_frames() -> None:
    model = make_bezier([all_frames()])
    instance = model.instances_for_test[0]
    for frame in range(NUM_FRAMES):
        means, _, valid = model.means_at(float(TIMESTAMPS[frame]))
        pose = instance["poses"][frame]
        expected = instance["pts"] @ pose[:3, :3].T + pose[:3, 3]
        assert valid.all()
        assert (means - expected).norm(dim=-1).max() < 0.05


def test_short_or_sparse_instances_are_skipped() -> None:
    one_frame = [False] * NUM_FRAMES
    one_frame[4] = True
    model = make_bezier([all_frames(), one_frame, all_frames()], num_pts=[40, 40, 10])
    assert model.num_points == 40
    assert model.traj_control_points.shape[0] == 1
    means, offsets, valid = model.means_at(float(TIMESTAMPS[4]))
    assert torch.isfinite(means).all() and torch.isfinite(offsets).all()
    assert valid.all()


def test_hidden_outside_exist_range_visible_in_gap() -> None:
    info = [False, False, True, True, True, False, False, True, True, False]
    model = make_bezier([info])
    for frame, expected_valid in [(0, False), (5, True), (9, False)]:
        _, _, valid = model.means_at(float(TIMESTAMPS[frame]))
        assert valid.all() == expected_valid
        model.set_cur_frame(frame)
        opacities = model.get_gaussians(fake_camera())["_opacities"]
        assert (opacities > 0).all() == expected_valid
        assert model.get_gaussians(fake_camera())["_means"].shape[0] == model.num_points


def test_velocity_matches_finite_difference_of_offsets() -> None:
    model = make_bezier([all_frames(), all_frames()])
    time = float(TIMESTAMPS[3]) + 0.02
    velocity = model.velocity_at(time)
    u = model.bezier_params(time)[model.point_ids[:, 0]][:, None]
    h = 1e-2

    def offset_at(param: "torch.Tensor") -> "torch.Tensor":
        basis = bm.bernstein(param, 3)
        return torch.einsum("nk,nkd->nd", basis, model._control_points)

    expected = (offset_at(u + h) - offset_at(u - h)) / (2 * h)
    assert torch.allclose(velocity, expected, atol=1e-3)
    outside = make_bezier([[False] * 3 + [True] * 3 + [False] * 4])
    assert (outside.velocity_at(float(TIMESTAMPS[9])) == 0).all()


def test_split_keeps_point_ids_and_control_points() -> None:
    model = make_bezier([all_frames(), all_frames()])
    mask = torch.zeros(model.num_points, dtype=torch.bool)
    mask[:3] = True
    parents = model._control_points[mask].detach().clone()
    scales_before = model._scales[mask].detach().clone()
    with torch.no_grad():  # split_gaussians sửa scale của cha tại chỗ, như refinement_after
        new_cp, _, _, _, new_scales, _, new_ids = model.split_gaussians(mask, 2)
    assert new_cp.shape == (6, 4, 3)
    assert new_ids.shape[0] == 6
    delta = new_cp - parents.repeat(2, 1, 1)
    assert torch.allclose(delta, delta[:, :1].expand_as(delta))
    assert torch.allclose(model._scales[mask], scales_before - torch.log(torch.tensor(1.6)))
    assert torch.allclose(new_scales, model._scales[mask].repeat(2, 1))


def test_refinement_keeps_all_shapes_and_optimizer_state_consistent() -> None:
    model = make_bezier([all_frames(), all_frames()])
    groups = model.get_param_groups()
    optimizer = torch.optim.Adam(
        [{"params": params, "name": name, "lr": 1e-3} for name, params in groups.items()]
    )
    for params in groups.values():
        params[0].grad = torch.zeros_like(params[0])
    optimizer.step()
    model.step = 1000
    model.xys_grad_norm = torch.ones(model.num_points)
    model.vis_counts = torch.ones(model.num_points)
    model.max_2Dsize = torch.zeros(model.num_points)
    before = model.num_points
    model.refinement_after(1000, optimizer)
    assert model.num_points > before
    assert model._control_points.shape == (model.num_points, 4, 3)
    assert model.point_ids.shape[0] == model.num_points
    for params in model.get_gaussian_param_groups().values():
        assert optimizer.state[params[0]]["exp_avg"].shape == params[0].shape


def test_param_group_names() -> None:
    model = make_bezier([all_frames()])
    names = ["xyz", "sh_dc", "sh_rest", "opacity", "scaling", "rotation", "traj", "bezier_param"]
    assert set(model.get_param_groups()) == {f"BezierNodes#{name}" for name in names}


def test_state_dict_round_trip_with_different_sizes() -> None:
    source = make_bezier([all_frames(), all_frames(), all_frames()])
    target = make_bezier([all_frames()])
    assert target.num_points != source.num_points
    target.load_state_dict(source.state_dict())
    assert target.num_points == source.num_points
    assert target.traj_control_points.shape == source.traj_control_points.shape
    for frame in (0, 4, 9):
        time = float(TIMESTAMPS[frame])
        assert torch.equal(target.means_at(time)[0], source.means_at(time)[0])
    fresh = BezierNodes(
        class_name="BezierNodes",
        ctrl=source.ctrl_cfg,
        device=torch.device("cpu"),
    )
    fresh.load_state_dict(source.state_dict())
    assert torch.equal(fresh.means_at(0.5)[0], source.means_at(0.5)[0])


def test_icc_loss_zero_for_constant_offset_norm() -> None:
    model = make_bezier([all_frames()])
    with torch.no_grad():
        model._control_points.copy_(model._control_points[:, :1].expand(-1, 4, -1))
    model.set_cur_frame(4)
    model.get_gaussians(fake_camera())
    assert model.icc_loss().abs() < 1e-5


def test_icc_loss_positive_when_offset_norm_changes_and_zero_without_valid() -> None:
    model = make_bezier([[False] * 6 + [True] * 4])
    model.set_cur_frame(8)
    model.get_gaussians(fake_camera())
    assert model.icc_loss() >= 0
    model.set_cur_frame(0)
    model.get_gaussians(fake_camera())
    assert model.icc_loss() == 0
