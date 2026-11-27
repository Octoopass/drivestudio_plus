"""Lưu trạng thái tối ưu của các node XSIM để worker có thể tiếp tục train.

Sau khi nạp trọng số, XSIM tạo Parameter mới cho các Gaussian. Worker gọi
`restore_optimization` để gắn optimizer vào các Parameter này rồi khôi phục
moment và lịch learning rate theo tên nhóm (kế hoạch tích hợp, mục A05).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xsim.modeling.nodes.common.node_collection import NodeCollection


def capture_optimization(nodes: "NodeCollection") -> dict:
    """Thu trạng thái Adam và số bước của từng scheduler theo tên node và nhóm."""
    return {
        name: {
            "optimizer": module.opt.state_dict(),
            "scheduler_steps": {s.param: s._step for s in module.schedulers},
        }
        for name, module in nodes.named_modules()
        if getattr(module, "opt", None) is not None
    }


def restore_optimization(nodes: "NodeCollection", saved: dict) -> None:
    """Tạo optimizer sau khi nạp trọng số và ghép trạng thái theo tên nhóm.

    Args:
        nodes: NodeCollection đã nạp đầy đủ trọng số của checkpoint.
        saved: Kết quả của capture_optimization ở bước train đã lưu.

    Raises:
        ValueError: Khi cấu trúc node, nhóm tham số hoặc scheduler thay đổi.
    """
    nodes.init_optimization()
    modules = {
        name: module
        for name, module in nodes.named_modules()
        if getattr(module, "opt", None) is not None
    }
    if modules.keys() != saved.keys():
        raise ValueError("XSIM checkpoint optimizer nodes differ")
    for name, module in modules.items():
        snapshot = saved[name]
        state = snapshot["optimizer"]
        groups = {group["name"]: group for group in state["param_groups"]}
        order = [group["name"] for group in module.opt.param_groups]
        if (
            set(order) != groups.keys()
            or len(order) != len(groups)
            or len(groups) != len(state["param_groups"])
        ):
            raise ValueError(f"XSIM checkpoint optimizer groups differ for {name}")
        module.opt.load_state_dict({**state, "param_groups": [groups[key] for key in order]})
        steps = snapshot["scheduler_steps"]
        if {s.param for s in module.schedulers} != steps.keys():
            raise ValueError(f"XSIM checkpoint schedulers differ for {name}")
        for scheduler in module.schedulers:
            scheduler._step = steps[scheduler.param]
