"""Trainer của BezierGS: thêm ba loss riêng của BezierGS lên ``MultiTrainer``.

Nguồn: BezierGS (ICCV 2025), LogosIntelligenceLab/BezierGS, giấy phép MIT, commit ``6df0a36``,
file ``train.py``. Port vào khung DriveStudio của drive2sim; chỗ khác bản gốc ghi ở
``docs/superpowers/specs/2026-11-20-beziergs-design.md`` mục 3.

File này chứa:
- ``collect_bezier_instances``: gộp vật của ``RigidNodes`` (xe) và ``DeformableNodes`` (người) từ
  dataset thành dữ liệu khởi tạo cho ``BezierNodes``.
- ``dynamic_render_loss``: loss ảnh chỉ vẽ vật động so với ảnh thật nhân mask động.
- ``static_velocity_loss``: phạt vận tốc của Gaussian ở pixel tĩnh.
- ``BezierTrainer``: ``MultiTrainer`` có thêm một lần vẽ vật động và một lần vẽ bản đồ vận tốc
  (có gradient) mỗi bước train, rồi cộng ba loss ``bezier_dynamic_loss``,
  ``bezier_velocity_loss``, ``bezier_icc_loss``.

Stage ``train.drivestudio`` chọn trainer này qua ``trainer.type`` trong
``drive2sim/configs/methods/beziergs.yaml``. Mask động lấy từ ``image_infos["dynamic_masks"]``,
nên dataset phải bật ``load_dynamic_mask``. Phần vẽ chưa chạy thật trên GPU.
"""

from collections.abc import Callable

import torch

from drive2sim.recon.trainers.scene_graph import MultiTrainer

# Chia bản đồ vận tốc cho alpha: chặn dưới để pixel gần như trong suốt không làm nổ giá trị
# (EPS của train.py gốc).
ALPHA_EPS = 1e-5


def collect_bezier_instances(dataset: object, **init_cfg: object) -> dict[int, dict]:
    """Gom vật động của dataset thành dữ liệu khởi tạo cho ``BezierNodes``.

    BezierGS mô hình hoá mọi vật động bằng cùng một loại node, nên gộp vật của ``RigidNodes`` (xe)
    và ``DeformableNodes`` (người, kể cả người có SMPL: ``exclude_smpl=False``).

    Args:
        dataset: ``DrivingDataset`` (hoặc object có ``get_init_objects`` cùng chữ ký).
        **init_cfg: Khối ``init`` của ``BezierNodes`` trong config (``instance_max_pts``,
            ``only_moving``, ``traj_length_thres``).

    Returns:
        Dict ``{id vật trong dataset: dữ liệu vật}`` như ``get_init_objects``, mỗi vật có
        ``node_type == "BezierNodes"``.
    """
    vehicles = dataset.get_init_objects(cur_node_type="RigidNodes", **init_cfg)
    humans = dataset.get_init_objects(
        cur_node_type="DeformableNodes", exclude_smpl=False, **init_cfg
    )
    instances = {**vehicles, **humans}
    for instance in instances.values():
        instance["node_type"] = "BezierNodes"
    return instances


def dynamic_render_loss(
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    gt_rgb: torch.Tensor,
    dynamic_mask: torch.Tensor,
    ssim_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_dssim: float,
) -> torch.Tensor:
    """Tính loss của ảnh chỉ vẽ vật động (chưa nhân trọng số).

    ``(1 − λ)·L1(pred, gt·mask) + λ·(1 − SSIM(pred, gt·mask)) + mean|alpha − mask|``: ảnh vẽ
    riêng vật động phải khớp ảnh thật ở vùng mask động và trong suốt ở ngoài mask (``train.py``
    gốc, ``lambda_dynamic_render``).

    Args:
        pred_rgb: Ảnh chỉ vẽ Gaussian vật động, ``[H, W, 3]``.
        pred_alpha: Alpha của ảnh đó, ``[H, W, 1]`` (hoặc ``[H, W]``).
        gt_rgb: Ảnh thật, ``[H, W, 3]``.
        dynamic_mask: Mask động ``[H, W]``, 1 ở pixel vật động.
        ssim_fn: Hàm SSIM nhận hai ảnh ``[1, 3, H, W]`` và trả scalar (``trainer.ssim``).
        lambda_dssim: Hệ số của thành phần SSIM, thuộc ``[0, 1]``.

    Returns:
        Scalar tensor.
    """
    masked_gt = gt_rgb * dynamic_mask[..., None]
    l1 = torch.abs(pred_rgb - masked_gt).mean()
    ssim = ssim_fn(pred_rgb.permute(2, 0, 1)[None], masked_gt.permute(2, 0, 1)[None])
    alpha_match = torch.abs(pred_alpha.reshape(dynamic_mask.shape) - dynamic_mask).mean()
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim) + alpha_match


def static_velocity_loss(velocity_map: torch.Tensor, dynamic_mask: torch.Tensor) -> torch.Tensor:
    """Tính trung bình ``|vận tốc|`` trên các pixel tĩnh (chưa nhân trọng số).

    Vận tốc chỉ đáng có ở pixel vật động; phạt vận tốc ở pixel tĩnh để Gaussian của vật động
    không chiếm vùng tĩnh (``lambda_velocity`` trong ``train.py`` gốc).

    Args:
        velocity_map: Bản đồ vận tốc đã chia alpha, ``[H, W, 3]``.
        dynamic_mask: Mask động ``[H, W]``, 1 ở pixel vật động.

    Returns:
        Scalar tensor; ``0`` nếu ảnh không có pixel tĩnh nào.
    """
    static = dynamic_mask < 0.5
    if not static.any():
        return velocity_map.new_zeros(())
    return velocity_map[static].abs().mean()


class BezierTrainer(MultiTrainer):
    """``MultiTrainer`` có thêm các loss của BezierGS khi config có ``BezierNodes``.

    Mỗi bước train, ngoài lần vẽ chính của ``MultiTrainer``, vẽ thêm hai lần có gradient bằng
    ``rasterize_features``: ảnh chỉ gồm Gaussian của ``BezierNodes`` và bản đồ vận tốc. Config
    không có ``BezierNodes`` (hoặc node bị xoá vì rỗng) thì chạy y như ``MultiTrainer``.
    """

    def forward(
        self,
        image_infos: dict[str, torch.Tensor],
        camera_infos: dict[str, torch.Tensor],
        novel_view: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Vẽ như ``MultiTrainer.forward`` và khi train thêm kết quả vẽ của vật động.

        Khi train có ``BezierNodes``, thêm ``bezier_rgb``, ``bezier_opacity``, ``velocity_map``
        vào kết quả.

        Args:
            image_infos: Thông tin ảnh và pixel của khung; cần ``dynamic_masks`` khi train có
                ``BezierNodes``.
            camera_infos: Thông tin camera.
            novel_view: ``True`` thì tắt tinh chỉnh camera, như ``MultiTrainer``.

        Returns:
            Dict kết quả của ``MultiTrainer.forward``, cộng ba khoá trên khi train.

        Raises:
            ValueError: Nếu train có ``BezierNodes`` mà ``image_infos`` không có mask động.
        """
        wants_bezier = (
            self.training
            and torch.is_grad_enabled()  # FastGS chấm điểm Gaussian bằng lần vẽ không gradient
            and "BezierNodes" in self.gaussian_classes
        )
        if wants_bezier:
            self._dynamic_mask(image_infos)  # báo lỗi trước khi tốn ba lần vẽ
        outputs = super().forward(image_infos, camera_infos, novel_view)
        if wants_bezier:
            self._render_bezier_outputs(outputs, image_infos)
        return outputs

    @staticmethod
    def _dynamic_mask(image_infos: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lấy mask động ``[H, W]`` của khung, hoặc báo lỗi rõ ràng nếu dataset không nạp.

        Raises:
            ValueError: Nếu ``image_infos`` không có ``dynamic_masks`` (``load_dynamic_mask`` tắt).
        """
        mask = image_infos.get("dynamic_masks")
        if mask is None:
            raise ValueError(
                "BezierGS cần mask vật động nhưng dữ liệu không có 'dynamic_masks'. "
                "Bật dataset.load_dynamic_mask: True trong config và chạy stage "
                "prep.masks_segformer với dynamic: true."
            )
        return mask

    def _render_bezier_outputs(
        self, outputs: dict[str, torch.Tensor], image_infos: dict[str, torch.Tensor]
    ) -> None:
        """Vẽ ảnh chỉ vật động và bản đồ vận tốc, ghi vào ``outputs`` (có gradient).

        Dùng Gaussian và camera của lần vẽ chính (``self._last_render_inputs``) nên không phải
        gom lại. Vận tốc của Background bằng 0; bản đồ vận tốc là vận tốc vẽ như màu rồi chia cho
        alpha của chính lần vẽ đó (như ``train.py`` gốc).
        """
        gs, cam, _ = self._last_render_inputs
        node = self.models["BezierNodes"]
        is_bezier = self.pts_labels == self.gaussian_classes["BezierNodes"]

        rgb, alpha = self.rasterize_features(gs, cam, gs.rgbs, opacity_mask=is_bezier)
        outputs["bezier_rgb"] = self.affine_transformation(rgb, image_infos)
        outputs["bezier_opacity"] = alpha

        velocity = torch.zeros_like(gs.means)
        velocity[is_bezier] = node.velocity_at(node.cur_time)
        velocity_image, velocity_alpha = self.rasterize_features(gs, cam, velocity)
        outputs["velocity_map"] = velocity_image / velocity_alpha.clamp_min(ALPHA_EPS)

    def compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        image_infos: dict[str, torch.Tensor],
        cam_infos: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Tính loss của ``MultiTrainer`` cộng ba loss BezierGS.

        Returns:
            Dict loss; thêm ``bezier_dynamic_loss``, ``bezier_velocity_loss``, ``bezier_icc_loss``
            khi có ``BezierNodes`` (đã nhân trọng số ``losses.bezier_*.w``).
        """
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        loss_dict.update(self._bezier_losses(outputs, image_infos))
        return loss_dict

    def _bezier_losses(
        self, outputs: dict[str, torch.Tensor], image_infos: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Tính ba loss của BezierGS đã nhân trọng số.

        Bỏ qua (trả dict rỗng) khi không có ``BezierNodes`` (cảnh không có vật động, node bị xoá)
        hoặc ``outputs`` không có kết quả vẽ của vật động (lúc đánh giá).

        Args:
            outputs: Kết quả của ``forward``.
            image_infos: Thông tin ảnh; ``pixels`` là ảnh thật, ``dynamic_masks`` là mask động.

        Returns:
            Dict ``bezier_dynamic_loss``, ``bezier_velocity_loss``, ``bezier_icc_loss``.

        Raises:
            ValueError: Nếu không có mask động (xem ``_dynamic_mask``).
        """
        if "BezierNodes" not in self.gaussian_classes or "bezier_rgb" not in outputs:
            return {}
        mask = BezierTrainer._dynamic_mask(image_infos)
        weights = self.losses_dict
        # λ của SSIM trong loss RGB: ssim.w / (rgb.w + ssim.w) (= 0,2 ở BezierGS gốc).
        lambda_dssim = weights.ssim.w / (weights.rgb.w + weights.ssim.w)
        dynamic = dynamic_render_loss(
            outputs["bezier_rgb"],
            outputs["bezier_opacity"],
            image_infos["pixels"],
            mask,
            self.ssim,
            lambda_dssim,
        )
        velocity = static_velocity_loss(outputs["velocity_map"], mask)
        icc = self.models["BezierNodes"].icc_loss()
        return {
            "bezier_dynamic_loss": weights.bezier_dynamic.w * dynamic,
            "bezier_velocity_loss": weights.bezier_velocity.w * velocity,
            "bezier_icc_loss": weights.bezier_icc.w * icc,
        }
