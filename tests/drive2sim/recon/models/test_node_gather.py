"""Kiểm tra cách lấy pose theo node của OmniRe sau khi đổi sang ``index_select`` (CPU).

So ``RigidNodes.transform_means``, ``transform_quats`` với tham chiếu tính bằng phép lấy theo chỉ
số kiểu cũ (``x[point_ids]``): output trùng, gradient theo pose của từng xe lệch ở mức làm tròn.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch3d")

from omegaconf import OmegaConf  # noqa: E402

from drive2sim.core.config import load_config  # noqa: E402
from drive2sim.recon.models.gaussians.basics import quat_mult, quat_to_rotmat  # noqa: E402
from drive2sim.recon.models.nodes import RigidNodes  # noqa: E402


def make_rigid() -> "RigidNodes":
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
    torch.manual_seed(0)
    instances = {}
    for i in range(3):
        poses = torch.eye(4).repeat(4, 1, 1)
        poses[:, :3, 3] = torch.randn(4, 3) * 5
        instances[i] = {
            "class_name": "Car",
            "pts": torch.randn(20 + i, 3),
            "colors": torch.rand(20 + i, 3),
            "poses": poses,
            "size": torch.tensor([4.0, 2.0, 1.5]),
            "frame_info": torch.ones(4, dtype=torch.bool),
            "num_pts": 20 + i,
        }
    model.create_from_pcd(instances)
    model.set_cur_frame(2)
    return model


def test_transform_matches_old_indexing() -> None:
    model = make_rigid()
    ids = model.point_ids[..., 0]
    weights = torch.rand(model.num_points, 3)

    means = model.transform_means(model._means)
    (means * weights).sum().backward()
    new_trans, new_quats = model.instances_trans.grad.clone(), model.instances_quats.grad.clone()
    model.zero_grad(set_to_none=True)

    rot = quat_to_rotmat(model.quat_act(model.instances_quats[model.cur_frame]))
    reference = torch.bmm(rot[ids], model._means.unsqueeze(-1)).squeeze(-1)
    reference = reference + model.instances_trans[model.cur_frame][ids]
    (reference * weights).sum().backward()
    assert torch.allclose(means, reference, atol=1e-6)
    assert torch.allclose(new_trans, model.instances_trans.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(new_quats, model.instances_quats.grad, rtol=1e-5, atol=1e-6)

    quats = model.transform_quats(model._quats)
    expected = quat_mult(
        model.quat_act(model.instances_quats[model.cur_frame][ids]), model.quat_act(model._quats)
    )
    assert torch.allclose(quats, expected, atol=1e-6)
