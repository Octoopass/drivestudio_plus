"""Kiểm tra ``RigidNodes.refinement_after`` khi lớp chỉ còn đúng một Gaussian (CPU).

Trên cảnh dài, FastGS prune gần hết Gaussian của ``RigidNodes`` (opacity khởi tạo 0,1 bằng đúng
``prune_min_opacity``), có lúc chỉ còn một điểm. Khi đó mask densify dài 1 bị ``.squeeze()`` thành
số vô hướng, ``_scales[mask]`` có thêm một chiều và ``split_gaussians`` báo lỗi ``repeat``
(run ``20261127-140512-omnire-95ca`` dừng ở bước 8400). File này giữ hồi quy cho lỗi đó.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch3d")

from omegaconf import OmegaConf  # noqa: E402

from drive2sim.core.config import load_config  # noqa: E402
from drive2sim.recon.models.nodes import RigidNodes  # noqa: E402


def make_rigid(num_kept: int) -> "RigidNodes":
    cfg = OmegaConf.create(load_config("drive2sim/configs/methods/omnire.yaml"))
    model_cfg = cfg.model.RigidNodes
    model_cfg.ctrl = OmegaConf.merge(cfg.trainer.gaussian_ctrl_general_cfg, model_cfg.ctrl)
    model_cfg.optim = OmegaConf.merge(cfg.trainer.gaussian_optim_general_cfg, model_cfg.optim)
    model = RigidNodes(
        **model_cfg,
        class_name="RigidNodes",
        scene_scale=30.0,
        scene_origin=torch.zeros(3),
        num_train_images=12,
        device=torch.device("cpu"),
    )
    poses = torch.eye(4).repeat(4, 1, 1)
    instances = {
        0: {
            "class_name": "Car",
            "pts": torch.randn(8, 3),
            "colors": torch.rand(8, 3),
            "poses": poses,
            "size": torch.tensor([4.0, 2.0, 1.5]),
            "frame_info": torch.ones(4, dtype=torch.bool),
            "num_pts": 8,
        }
    }
    model.create_from_pcd(instances)
    # create_from_pcd cần vài điểm để khởi tạo scale bằng kNN; cắt còn num_kept điểm
    # như cull_gaussians
    keep = torch.zeros(model.num_points, dtype=torch.bool)
    keep[:num_kept] = True
    for name in ("_means", "_scales", "_quats", "_features_dc", "_features_rest", "_opacities"):
        setattr(model, name, torch.nn.Parameter(getattr(model, name)[keep].detach()))
    model.point_ids = model.point_ids[keep]
    model.set_cur_frame(2)
    return model


def test_refinement_after_survives_single_gaussian() -> None:
    model = make_rigid(num_kept=1)
    assert model.num_points == 1
    groups = [
        {"params": params, "name": name} for name, params in model.get_param_groups().items()
    ]
    optimizer = torch.optim.Adam(groups, lr=1e-3)
    # một bước tối ưu để optimizer có exp_avg, exp_avg_sq như trong lúc train thật
    sum(p.sum() for group in groups for p in group["params"]).backward()
    optimizer.step()
    ctrl = model.ctrl_cfg
    # bước densify hợp lệ: sau warmup, trước stop_split_at, đủ xa lần reset opacity gần nhất
    step = ctrl.reset_alpha_interval + max(12, ctrl.refine_interval) + 1
    assert ctrl.warmup_steps < step < ctrl.stop_split_at
    model.step = step
    model.xys_grad_norm = torch.full((1,), 1.0)
    model.vis_counts = torch.ones(1)
    model.max_2Dsize = torch.zeros(1)

    model.refinement_after(step, optimizer)

    # điểm duy nhất có thể bị cull (opacity, kích thước): chỉ cần không lỗi và các tensor còn khớp
    count = model.num_points
    assert model._scales.shape[0] == model._quats.shape[0] == model.point_ids.shape[0] == count


def test_reg_loss_is_finite_when_class_has_no_gaussians() -> None:
    """Lớp về 0 điểm thì loss regularization không được NaN (mean trên tensor rỗng)."""
    model = make_rigid(num_kept=0)
    assert model.num_points == 0
    sharp = model.reg_cfg.get("sharp_shape_reg", None)
    # bước chia hết step_interval để nhánh sharp_shape_reg chạy
    model.step = 1000 * (sharp.step_interval if sharp is not None else 1)

    loss_dict = model.compute_reg_loss()

    for name, value in loss_dict.items():
        assert torch.isfinite(value).all(), name
