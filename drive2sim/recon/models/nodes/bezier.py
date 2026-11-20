"""Node Gaussian vật động theo đường Bézier của BezierGS, thay ``RigidNodes`` trong trainer.

Nguồn: BezierGS (ICCV 2025), LogosIntelligenceLab/BezierGS, giấy phép MIT, commit ``6df0a36``,
file ``scene/bz_gaussian_model.py``. Port vào khung DriveStudio của drive2sim; các chỗ khác bản
gốc ghi ở ``docs/superpowers/specs/2026-11-20-beziergs-design.md`` mục 3.

File này chứa:
- ``BezierNodes``: mỗi vật động là một nhóm Gaussian. Tâm nhóm đi theo đường Bézier bậc ``order``
  của vật (``traj_control_points``); mỗi Gaussian có độ lệch so với tâm cũng là một đường Bézier
  riêng (``_control_points``); thời gian thật đổi sang tham số ``u`` của Bézier qua một đa thức
  Bernstein học được của từng vật (``time_control_points``). Ngoài khoảng ``[khung đầu, khung
  cuối]`` vật có box thì nhóm bị ẩn (opacity bằng 0).

Trainer (``drive2sim/recon/trainers/bezier.py``) tạo node qua ``MultiTrainer``, gọi
``create_from_pcd`` với dữ liệu instance của ``DrivingDataset.get_init_objects``, gọi
``set_cur_frame`` trước mỗi lần render và dùng ``velocity_at``, ``icc_loss`` cho loss của
BezierGS. Toán Bézier nằm ở ``bezier_math.py``.

Không hỗ trợ các hàm chỉnh sửa instance và xuất PLY của ``RigidNodes`` (``remove_instances``,
``replace_instances``, ``export_gaussians_to_ply``, ``get_instance_activated_gs_dict``): chúng
đọc ``_means`` mà node này không có (``_means`` bị xoá để gọi nhầm thì báo lỗi ngay).
``ctrl.cull_out_of_bound`` cũng bị bỏ qua vì Gaussian không còn toạ độ trong hệ box.
"""

import logging
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.nn import Parameter

from drive2sim.recon.models.gaussians.basics import (
    RGB2SH,
    dataclass_camera,
    dup_in_optim,
    k_nearest_sklearn,
    num_sh_bases,
    quat_to_rotmat,
    remove_from_optim,
    spherical_harmonics,
)
from drive2sim.recon.models.gaussians.vanilla import VanillaGaussians
from drive2sim.recon.models.nodes.bezier_math import (
    bernstein,
    bernstein_derivative,
    fit_control_points,
    fit_time_map,
    fit_trajectory,
)

logger = logging.getLogger()

# Tham số mặc định của khối config ``bezier`` (spec beziergs mục 2).
DEFAULT_BEZIER_CONFIG = {
    "order": 3,
    "min_instance_pts": 30,
    "opacity_init": 0.01,
    "fit_rounds": 100,
}

# Tên các Parameter có số hàng bằng số Gaussian (để cấp phát lại khi nạp checkpoint).
_GAUSSIAN_PARAM_NAMES = (
    "_control_points",
    "_features_dc",
    "_features_rest",
    "_opacities",
    "_scales",
    "_quats",
)


class BezierNodes(VanillaGaussians):
    """Các Gaussian của vật động, chuyển động theo đường Bézier học được (BezierGS).

    Một node chứa mọi vật động của một loại (xe và người gộp chung, xem
    ``collect_bezier_instances`` ở trainer). Mỗi Gaussian thuộc một vật (``point_ids``); vị trí
    world ở thời điểm ``t`` là ``tâm vật(u) + độ lệch Gaussian(u)`` với ``u`` là tham số Bézier
    của vật ở thời điểm đó (``bezier_params``).

    Attributes:
        order: Bậc Bézier (3 nghĩa là 4 control point).
        min_instance_pts: Vật có ít điểm hơn số này bị bỏ khi khởi tạo.
        opacity_init: Opacity khởi tạo của mọi Gaussian (trước logit).
        fit_rounds: Số vòng lặp tối đa khi fit đường đi của vật.
        _control_points: Control point độ lệch của từng Gaussian so với tâm vật, hệ world, mét,
            ``Parameter [N, K, 3]`` (K = order + 1).
        traj_control_points: Control point đường đi tâm vật, ``Parameter [I, K, 3]``.
        time_control_points: Hệ số đa thức đổi thời gian chuẩn hoá sang ``u``,
            ``Parameter [I, K, 1]``.
        point_ids: Chỉ số vật trong node của từng Gaussian, ``LongTensor [N, 1]``.
        exist_range: Thời gian chuẩn hoá của khung đầu và khung cuối có box, ``[I, 2]``.
        instances_fv: Vật có box ở khung nào, ``BoolTensor [F, I]``.
        instances_size: Kích thước box của từng vật, ``[I, 3]``.
        cur_frame: Khung đang render (``set_cur_frame``).
        cur_time: Thời gian chuẩn hoá của khung đang render.
        cur_offsets: Độ lệch của từng Gaussian ở lần ``get_gaussians`` gần nhất, ``[N, 3]``.
        cur_valid: Gaussian có hiện ở lần ``get_gaussians`` gần nhất, ``BoolTensor [N]``.
    """

    def __init__(self, bezier: Optional[OmegaConf] = None, **kwargs: object) -> None:
        """Dựng node rỗng; ``create_from_pcd`` hoặc ``load_state_dict`` điền dữ liệu sau.

        Args:
            bezier: Khối config ``bezier`` với các khoá ``order``, ``min_instance_pts``,
                ``opacity_init``, ``fit_rounds``. ``None`` dùng ``DEFAULT_BEZIER_CONFIG``.
            **kwargs: Các tham số của ``VanillaGaussians`` (``class_name``, ``ctrl``, ``reg``,
                ``device``, ...); khoá ``init``, ``optim``, ``type`` được bỏ qua như ở
                ``RigidNodes``.
        """
        super().__init__(**kwargs)
        cfg = OmegaConf.merge(OmegaConf.create(DEFAULT_BEZIER_CONFIG), bezier or {})
        self.order = int(cfg.order)
        self.min_instance_pts = int(cfg.min_instance_pts)
        self.opacity_init = float(cfg.opacity_init)
        self.fit_rounds = int(cfg.fit_rounds)
        num_cp = self.order + 1
        # Node này không có ``_means``; xoá để mã thừa kế đọc nhầm sẽ lỗi ngay thay vì âm thầm
        # dùng tensor giả của VanillaGaussians.
        del self._means
        self._control_points = torch.zeros(1, num_cp, 3, device=self.device)
        self.traj_control_points = torch.zeros(0, num_cp, 3, device=self.device)
        self.time_control_points = torch.zeros(0, num_cp, 1, device=self.device)
        self.point_ids = torch.zeros(0, 1, dtype=torch.long, device=self.device)
        self.exist_range = torch.zeros(0, 2, device=self.device)
        self.instances_fv = torch.zeros(0, 0, dtype=torch.bool, device=self.device)
        self.instances_size = torch.zeros(0, 3, device=self.device)
        self.cur_frame = 0
        self.cur_time = 0.0
        self.cur_offsets: Optional[torch.Tensor] = None
        self.cur_valid: Optional[torch.Tensor] = None

    @property
    def num_points(self) -> int:
        """Số Gaussian của node."""
        return self._control_points.shape[0]

    @property
    def num_instances(self) -> int:
        """Số vật trong node."""
        return self.traj_control_points.shape[0]

    @property
    def num_frames(self) -> int:
        """Số khung của cảnh."""
        return self.instances_fv.shape[0]

    def set_cur_frame(self, frame_id: int) -> None:
        """Đặt khung sẽ render và thời gian chuẩn hoá tương ứng (``cur_time``)."""
        self.cur_frame = frame_id
        self.cur_time = float(self.normalized_timestamps[frame_id])

    def register_normalized_timestamps(self, normalized_timestamps: torch.Tensor) -> None:
        """Nhận thời gian chuẩn hoá ``[0, 1]`` của các khung từ trainer."""
        self.normalized_timestamps = normalized_timestamps

    def create_from_pcd(self, instance_pts_dict: dict[int, dict[str, object]]) -> None:
        """Khởi tạo Gaussian, đường đi và hàm map thời gian từ dữ liệu instance của dataset.

        Mỗi vật: lấy các khung có box, fit đường đi tâm bằng Bézier, đổi điểm trong hệ vật sang
        độ lệch world ở từng khung (``R_t · p``) rồi fit control point độ lệch theo cùng tham số
        ``u`` của các khung đó. Vật chỉ có 1 khung có box, quá ít điểm hoặc fit ra giá trị không
        hữu hạn thì bị bỏ (log cảnh báo). Thứ tự vật trong node theo thứ tự trong dict, sau khi
        bỏ.

        Args:
            instance_pts_dict: Dạng của ``DrivingDataset.get_init_objects``: mỗi vật có ``pts``
                ``[n, 3]`` trong hệ vật, ``colors`` ``[n, 3]``, ``poses`` ``[F, 4, 4]`` (vật sang
                world), ``size`` ``[3]``, ``frame_info`` ``[F]`` bool và ``num_pts``.
        """
        timestamps = self.normalized_timestamps.to(self.device)
        control_points, colors, point_ids = [], [], []
        traj_cps, time_cps, ranges, visible, sizes = [], [], [], [], []
        for dataset_id, instance in instance_pts_dict.items():
            frame_info = instance["frame_info"].bool().to(self.device)
            frames = torch.where(frame_info)[0]
            num_pts = instance["pts"].shape[0]
            if frames.numel() < 2 or num_pts < self.min_instance_pts:
                logger.warning(
                    f"BezierNodes: bỏ vật {dataset_id} ({frames.numel()} khung có box, "
                    f"{num_pts} điểm; cần ít nhất 2 khung và {self.min_instance_pts} điểm)"
                )
                continue
            pts = instance["pts"].to(self.device).float()
            poses = instance["poses"].to(self.device).float()[frames]
            times = timestamps[frames]
            traj_cp, bezier_t = fit_trajectory(poses[:, :3, 3], self.order, rounds=self.fit_rounds)
            # Độ lệch của điểm p ở khung t là R_t · p (BezierGS, create_from_ply_dict).
            offsets = torch.einsum("tij,nj->nti", poses[:, :3, :3], pts)
            offset_cp = fit_control_points(offsets, bezier_t[:, None], self.order)
            time_cp = fit_time_map(times, bezier_t, self.order)
            if not all(torch.isfinite(x).all() for x in (traj_cp, offset_cp, time_cp)):
                logger.warning(f"BezierNodes: bỏ vật {dataset_id} vì fit ra giá trị không hữu hạn")
                continue
            instance_index = len(traj_cps)
            control_points.append(offset_cp)
            colors.append(instance["colors"].to(self.device).float())
            point_ids.append(torch.full((num_pts, 1), instance_index, dtype=torch.long))
            traj_cps.append(traj_cp)
            time_cps.append(time_cp)
            ranges.append(torch.stack([times[0], times[-1]]))
            visible.append(frame_info.unsqueeze(1))
            sizes.append(instance["size"].to(self.device).float())

        num_cp = self.order + 1
        if not traj_cps:
            self.traj_control_points = Parameter(torch.zeros(0, num_cp, 3, device=self.device))
            self.time_control_points = Parameter(torch.zeros(0, num_cp, 1, device=self.device))
            self._control_points = Parameter(torch.zeros(0, num_cp, 3, device=self.device))
            self.point_ids = torch.zeros(0, 1, dtype=torch.long, device=self.device)
            self.exist_range = torch.zeros(0, 2, device=self.device)
            self.instances_fv = torch.zeros(
                len(timestamps), 0, dtype=torch.bool, device=self.device
            )
            self.instances_size = torch.zeros(0, 3, device=self.device)
            self._set_empty_gaussian_attributes()
            return

        self.traj_control_points = Parameter(torch.stack(traj_cps))
        self.time_control_points = Parameter(torch.stack(time_cps))
        self.exist_range = torch.stack(ranges)
        self.instances_fv = torch.cat(visible, dim=1)
        self.instances_size = torch.stack(sizes)
        self.point_ids = torch.cat(point_ids, dim=0).to(self.device)
        self._control_points = Parameter(torch.cat(control_points, dim=0))
        init_colors = torch.cat(colors, dim=0)

        # Scale ban đầu: trung bình khoảng cách tới 3 láng giềng gần nhất, tính riêng từng vật
        # trên control point đầu (như distCUDA2 từng vật của bản gốc).
        avg_dists = []
        for instance_index in range(len(traj_cps)):
            own = self._control_points.data[self.point_ids[:, 0] == instance_index][:, 0, :]
            distances, _ = k_nearest_sklearn(own, 3)
            avg_dists.append(torch.from_numpy(distances).mean(dim=-1, keepdim=True))
        avg_dist = torch.cat(avg_dists, dim=0).to(self.device).clamp(0.002, 100)
        self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        # Quaternion đơn vị: Gaussian của vật động không xoay theo vật (như bản gốc).
        quats = torch.zeros(self.num_points, 4, device=self.device)
        quats[:, 0] = 1.0
        self._quats = Parameter(quats)

        dim_sh = num_sh_bases(self.sh_degree)
        shs = torch.zeros((self.num_points, dim_sh, 3), device=self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(init_colors)
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        opacity = torch.logit(torch.full((self.num_points, 1), self.opacity_init))
        self._opacities = Parameter(opacity.to(self.device))

    def _set_empty_gaussian_attributes(self) -> None:
        """Đặt các thuộc tính Gaussian về rỗng khi không có vật nào hợp lệ."""
        dim_rest = num_sh_bases(self.sh_degree) - 1
        self._scales = Parameter(torch.zeros(0, 3, device=self.device))
        self._quats = Parameter(torch.zeros(0, 4, device=self.device))
        self._features_dc = Parameter(torch.zeros(0, 3, device=self.device))
        self._features_rest = Parameter(torch.zeros(0, dim_rest, 3, device=self.device))
        self._opacities = Parameter(torch.zeros(0, 1, device=self.device))

    def bezier_params(self, time: float) -> torch.Tensor:
        """Đổi thời gian chuẩn hoá sang tham số ``u`` của Bézier cho từng vật.

        Args:
            time: Thời gian chuẩn hoá của khung, thuộc ``[0, 1]``.

        Returns:
            Tensor ``[I]``, ``u`` của từng vật (có thể ngoài ``[0, 1]`` khi ``time`` ngoài khoảng
            xuất hiện của vật; khi đó Gaussian của vật bị ẩn nên không ảnh hưởng ảnh).
        """
        basis = bernstein(torch.tensor([[time]], device=self.device), self.order)  # [1, K]
        return (basis @ self.time_control_points).reshape(-1)

    def means_at(self, time: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tính vị trí world, độ lệch và độ hiện của mọi Gaussian ở thời điểm ``time``.

        Args:
            time: Thời gian chuẩn hoá của khung, thuộc ``[0, 1]``.

        Returns:
            ``(means [N, 3], offsets [N, 3], valid [N] bool)``: ``means = tâm vật + offsets``
            (mét, world); ``valid`` đúng khi ``time`` nằm trong
            ``[exist_range[id, 0], exist_range[id, 1]]`` của vật (kể cả khoảng giữa mà vật bị che
            và không có box, vị trí khi đó nội suy theo Bézier như bản gốc).
        """
        u = self.bezier_params(time)  # [I]
        basis = bernstein(u[:, None], self.order)  # [I, K]
        ids = self.point_ids[:, 0]
        offsets = torch.einsum("nk,nkd->nd", basis[ids], self._control_points)
        centers = torch.einsum("ik,ikd->id", basis, self.traj_control_points)  # [I, 3]
        in_range = (self.exist_range[:, 0] <= time) & (time <= self.exist_range[:, 1])
        return offsets + centers[ids], offsets, in_range[ids]

    def velocity_at(self, time: float) -> torch.Tensor:
        """Tính vận tốc của từng Gaussian: đạo hàm theo ``u`` của đường Bézier độ lệch.

        Chỉ tính phần độ lệch, không tính phần tâm vật, như ``get_inst_velocity`` của bản gốc;
        loss vận tốc dùng nó để ép Gaussian ở pixel tĩnh không có chuyển động riêng.

        Args:
            time: Thời gian chuẩn hoá của khung.

        Returns:
            Tensor ``[N, 3]``; bằng 0 ở Gaussian không hiện (``valid`` sai).
        """
        u = self.bezier_params(time)
        derivative = bernstein_derivative(u[:, None], self.order)  # [I, K]
        ids = self.point_ids[:, 0]
        velocity = torch.einsum("nk,nkd->nd", derivative[ids], self._control_points)
        in_range = (self.exist_range[:, 0] <= time) & (time <= self.exist_range[:, 1])
        return velocity * in_range[ids].unsqueeze(-1).to(velocity.dtype)

    def get_gaussians(self, cam: dataclass_camera) -> dict[str, torch.Tensor]:
        """Tính Gaussian world ở khung hiện tại (``cur_time``) cho renderer.

        Giữ nguyên số Gaussian (Gaussian không hiện chỉ bị nhân opacity với 0) để nhãn class
        trong trainer khớp. Lưu ``cur_offsets``, ``cur_valid`` cho ``icc_loss``.

        Args:
            cam: Camera đang render; dùng vị trí camera để tính màu SH theo hướng nhìn.

        Returns:
            Dict cùng khoá với ``VanillaGaussians.get_gaussians``: ``_means``, ``_opacities``,
            ``_rgbs``, ``_scales``, ``_quats``.

        Raises:
            ValueError: Nếu có NaN hoặc Inf trong kết quả.
        """
        self.filter_mask = torch.ones(self.num_points, dtype=torch.bool, device=self.device)
        world_means, offsets, valid = self.means_at(self.cur_time)
        self.cur_offsets, self.cur_valid = offsets, valid

        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            degree = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = torch.clamp(spherical_harmonics(degree, viewdirs, colors) + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        gs_dict = dict(
            _means=world_means,
            _opacities=self.get_opacity * valid.unsqueeze(-1).to(world_means.dtype),
            _rgbs=rgbs,
            _scales=self.get_scaling,
            _quats=self.get_quats,
        )
        for name, value in gs_dict.items():
            if torch.isnan(value).any():
                raise ValueError(f"NaN detected in gaussian {name} at step {self.step}")
            if torch.isinf(value).any():
                raise ValueError(f"Inf detected in gaussian {name} at step {self.step}")
        return gs_dict

    def icc_loss(self) -> torch.Tensor:
        """Tính loss ICC của BezierGS: giữ độ dài độ lệch gần chuẩn của từng Gaussian.

        ``|(‖cp₀‖ + ‖cp_{K−1}‖)/2 − ‖độ lệch hiện tại‖|`` trung bình trên Gaussian đang hiện, với
        độ lệch hiện tại lấy từ lần ``get_gaussians`` gần nhất.

        Returns:
            Scalar tensor; ``0`` nếu chưa gọi ``get_gaussians`` hoặc không Gaussian nào hiện.
        """
        if self.cur_offsets is None or self.cur_valid is None or not self.cur_valid.any():
            return torch.zeros((), device=self.device)
        control_points = self._control_points[self.cur_valid]
        standard = 0.5 * (control_points[:, 0].norm(dim=-1) + control_points[:, -1].norm(dim=-1))
        return (standard - self.cur_offsets[self.cur_valid].norm(dim=-1)).abs().mean()

    def get_gaussian_param_groups(self) -> dict[str, list[Parameter]]:
        """Nhóm tham số của Gaussian (``xyz`` là control point độ lệch)."""
        return {
            self.class_prefix + "xyz": [self._control_points],
            self.class_prefix + "sh_dc": [self._features_dc],
            self.class_prefix + "sh_rest": [self._features_rest],
            self.class_prefix + "opacity": [self._opacities],
            self.class_prefix + "scaling": [self._scales],
            self.class_prefix + "rotation": [self._quats],
        }

    def get_param_groups(self) -> dict[str, list[Parameter]]:
        """Nhóm tham số của Gaussian cộng đường đi (``traj``) và hàm map thời gian."""
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix + "traj"] = [self.traj_control_points]
        param_groups[self.class_prefix + "bezier_param"] = [self.time_control_points]
        return param_groups

    def refinement_after(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        """Densify (split, nhân đôi), cull và đặt lại opacity; chép theo ``RigidNodes``.

        Khác ``RigidNodes``: thao tác trên ``_control_points`` ``[N, K, 3]`` thay cho ``_means``
        và không cull theo box (xem docstring của file).

        Args:
            step: Bước train hiện tại; phải bằng ``self.step``.
            optimizer: Optimizer có các nhóm tham số của ``get_param_groups``.
        """
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = self.step < self.ctrl_cfg.stop_split_at and self.step % (
                reset_interval
            ) > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            print(
                f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}"
            )
            if do_densification:
                assert (
                    self.xys_grad_norm is not None
                    and self.vis_counts is not None
                    and self.max_2Dsize is not None
                )
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                high_grads = self.fastgs_filter_densify(high_grads)

                size_thresh = self.ctrl_cfg.densify_size_thresh * self.scene_scale
                splits = (self.get_scaling.max(dim=-1).values > size_thresh).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                num_samples = self.ctrl_cfg.n_split_samples
                split_attrs = self.split_gaussians(splits, num_samples)

                dups = (self.get_scaling.max(dim=-1).values <= size_thresh).squeeze()
                dups &= high_grads
                dup_attrs = self.dup_gaussians(dups)

                names = list(_GAUSSIAN_PARAM_NAMES)
                for name, split_value, dup_value in zip(names, split_attrs, dup_attrs):
                    old = getattr(self, name).detach()
                    setattr(self, name, Parameter(torch.cat([old, split_value, dup_value], dim=0)))
                self.point_ids = torch.cat([self.point_ids, split_attrs[-1], dup_attrs[-1]], dim=0)
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_attrs[4][:, 0]),
                        torch.zeros_like(dup_attrs[4][:, 0]),
                    ],
                    dim=0,
                )
                dup_in_optim(
                    optimizer,
                    torch.where(splits)[0],
                    self.get_gaussian_param_groups(),
                    n=num_samples,
                )
                dup_in_optim(optimizer, torch.where(dups)[0], self.get_gaussian_param_groups(), 1)

            # Cull và đặt lại opacity lệch refine_interval để không lưu checkpoint đúng lúc reset.
            if self.step % reset_interval > max(
                self.num_train_images, self.ctrl_cfg.refine_interval
            ):
                deleted_mask = self.cull_gaussians()
                remove_from_optim(optimizer, deleted_mask, self.get_gaussian_param_groups())
            print(f"Class {self.class_prefix} left points: {self.num_points}")

            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                reset_value = torch.min(
                    self.get_opacity.data,
                    torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value,
                )
                self._opacities.data = torch.logit(reset_value)
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix + "opacity":
                        state = optimizer.state[group["params"][0]]
                        state["exp_avg"] = torch.zeros_like(state["exp_avg"])
                        state["exp_avg_sq"] = torch.zeros_like(state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self) -> torch.Tensor:
        """Xoá Gaussian có opacity thấp, scale quá lớn hoặc chiếm quá nhiều màn hình.

        Returns:
            Mask bool ``[N]`` (kích thước trước khi xoá) của các Gaussian đã xoá, để xoá trạng
            thái optimizer tương ứng.
        """
        before = self.num_points
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            too_big = (
                torch.exp(self._scales).max(dim=-1).values
                > self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | too_big
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        culls = self.fastgs_extra_culls(culls)
        for name in _GAUSSIAN_PARAM_NAMES:
            setattr(self, name, Parameter(getattr(self, name)[~culls].detach()))
        self.point_ids = self.point_ids[~culls]
        print(f"     Cull: {before - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int = 2) -> tuple:
        """Tách Gaussian quá lớn: thêm ``samps`` con quanh mỗi cha và thu nhỏ cha.

        Mỗi con lấy một mẫu ngẫu nhiên theo scale và quaternion của cha rồi cộng cùng một mẫu
        vào cả K control point của cha, nên cả đường Bézier độ lệch dịch rigid theo mẫu đó. Scale
        của cha và con chia cho 1,6 (như bản gốc).

        Args:
            split_mask: Mask bool ``[N]`` của Gaussian cần tách.
            samps: Số con mỗi cha.

        Returns:
            Tuple ``(control_points, features_dc, features_rest, opacities, scales, quats,
            point_ids)`` của các Gaussian mới, theo thứ tự các thuộc tính của
            ``_GAUSSIAN_PARAM_NAMES`` cộng ``point_ids`` ở cuối.
        """
        print(f"    Split: {split_mask.sum().item()}")
        samples = self.get_scaling[split_mask].repeat(samps, 1) * torch.randn(
            (samps * int(split_mask.sum().item()), 3), device=self.device
        )
        rots = quat_to_rotmat(self.get_quats[split_mask].repeat(samps, 1))
        shift = torch.bmm(rots, samples[..., None]).squeeze(-1)  # [M, 3]
        new_control_points = self._control_points[split_mask].repeat(samps, 1, 1) + shift[:, None]
        size_factor = 1.6
        shrunk = torch.log(torch.exp(self._scales[split_mask]) / size_factor)
        self._scales[split_mask] = shrunk
        return (
            new_control_points,
            self._features_dc[split_mask].repeat(samps, 1),
            self._features_rest[split_mask].repeat(samps, 1, 1),
            self._opacities[split_mask].repeat(samps, 1),
            shrunk.repeat(samps, 1),
            self._quats[split_mask].repeat(samps, 1),
            self.point_ids[split_mask].repeat(samps, 1),
        )

    def dup_gaussians(self, dup_mask: torch.Tensor) -> tuple:
        """Nhân đôi Gaussian nhỏ có gradient cao; thứ tự trả về như ``split_gaussians``."""
        print(f"      Dup: {dup_mask.sum().item()}")
        return (
            self._control_points[dup_mask],
            self._features_dc[dup_mask],
            self._features_rest[dup_mask],
            self._opacities[dup_mask],
            self._scales[dup_mask],
            self._quats[dup_mask],
            self.point_ids[dup_mask],
        )

    def state_dict(self) -> dict:
        """Trả state của các Parameter cộng các tensor không phải Parameter của node."""
        state_dict = super().state_dict()
        state_dict.update(
            {
                "point_ids": self.point_ids,
                "exist_range": self.exist_range,
                "instances_fv": self.instances_fv,
                "instances_size": self.instances_size,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: dict, **kwargs: object) -> str:
        """Nạp state; cấp phát mọi tham số theo shape trong ``state_dict`` rồi nạp giá trị.

        Số Gaussian và số vật lúc lưu có thể khác lúc khởi tạo (densify, cull, bỏ vật), nên
        không dùng shape hiện có của node. Không gọi ``VanillaGaussians.load_state_dict`` vì nó
        đọc ``_means``.

        Args:
            state_dict: Kết quả của ``state_dict()`` (được sửa tại chỗ: các khoá không phải
                Parameter bị lấy ra).
            **kwargs: Chuyển cho ``torch.nn.Module.load_state_dict`` (ví dụ ``strict``).

        Returns:
            Thông báo của ``torch.nn.Module.load_state_dict``.
        """
        self.point_ids = state_dict.pop("point_ids").to(self.device)
        self.exist_range = state_dict.pop("exist_range").to(self.device)
        self.instances_fv = state_dict.pop("instances_fv").to(self.device)
        self.instances_size = state_dict.pop("instances_size").to(self.device)
        for name in (*_GAUSSIAN_PARAM_NAMES, "traj_control_points", "time_control_points"):
            setattr(self, name, Parameter(torch.zeros(state_dict[name].shape, device=self.device)))
        return super(VanillaGaussians, self).load_state_dict(state_dict, **kwargs)
