"""Kiểm tra trước khi chạy run trên Colab, cho Studio và người dùng CLI (như ``kaggle/setup.py``).

Kiểm tra gồm: lệnh ``colab`` đã cài chưa, đăng nhập Google được không (``colab sessions``, chỉ
đọc), số compute unit còn lại (``colab usage``), có file SMPL trên máy để upload không (chỉ cần với
run dựng người), và commit hiện tại đã push chưa (VM ``git clone`` code từ remote).

Colab không cho biết trước còn GPU hay không: tài khoản miễn phí thường được T4 nhưng không bảo
đảm, và chỉ biết khi tạo phiên (``colab new`` báo lỗi khi hết chỗ hoặc không được cấp GPU đó). Vì
vậy ``gpu`` chỉ báo các phiên GPU đang có và số compute unit, không chặn ``ready``.

Đăng nhập không làm được từ Studio: lệnh ``colab`` mở trang đồng ý của Google ở lần gọi đầu, nên
người dùng chạy ``colab sessions`` một lần trong terminal.

File này chứa ``check``.
"""

import os
from pathlib import Path
from typing import Optional

from drive2sim import paths
from drive2sim.colab.client import GPUS, ColabClient, ColabError
from drive2sim.colab.executor import code_source
from drive2sim.engine import remote

KEYRING_HINT = (
    "máy không có keyring của hệ điều hành (server không màn hình): cài google-colab-cli kèm "
    "keyrings.alt và đặt PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring"
)


def check(client: Optional[ColabClient] = None, workdir_root: Optional[Path] = None) -> dict:
    """Kiểm tra Colab đã sẵn sàng để chạy run chưa.

    Mỗi bước độc lập, để người dùng thấy hết những gì còn thiếu một lần.

    Args:
        client: Client Colab; mặc định dùng lệnh ``colab`` thật.
        workdir_root: Thư mục làm việc, để tìm file SMPL ở ``<thư mục làm việc>/smpl_models/``.

    Returns:
        ``{"ready", "cli": {"installed", "version"}, "account": {"ok", "error", "hint"},
        "gpu": {"sessions": [{"name", "hardware"}], "balance", "rate_per_h", "choices"},
        "smpl": {"found", "path"}, "code": {"ok", "commit", "error", "dirty"}}``. ``ready`` đúng
        khi đã cài lệnh, đăng nhập được và code đã push. ``hint`` là gợi ý cho lỗi keyring.
    """
    client = client or ColabClient()
    version = client.version()
    account: dict[str, object] = {"ok": False, "error": None, "hint": None}
    gpu: dict[str, object] = {"sessions": [], "balance": None, "rate_per_h": None}
    gpu["choices"] = list(GPUS)
    if version is None:
        account["error"] = "chưa cài lệnh colab; cài bằng: uv tool install google-colab-cli"
    else:
        try:
            gpu["sessions"] = [s for s in client.sessions() if s["hardware"] in GPUS]
            account["ok"] = True
        except ColabError as error:
            account["error"] = f"{error}; đăng nhập bằng cách chạy colab sessions trong terminal"
            if "keyring" in str(error).lower() and not os.environ.get("PYTHON_KEYRING_BACKEND"):
                account["hint"] = KEYRING_HINT
        if account["ok"]:
            try:
                usage = client.usage()
                gpu["balance"], gpu["rate_per_h"] = usage["balance"], usage["rate_per_h"]
            except ColabError:
                pass  # chỉ là thông tin thêm
    try:
        commit, _ = code_source()
        code = {"ok": True, "commit": commit[:7], "error": None}
    except ColabError as error:
        code = {"ok": False, "commit": None, "error": str(error)}
    # VM clone đúng commit này: thay đổi chưa commit trên máy sẽ không có trên Colab.
    code["dirty"] = bool(remote.git("status", "--porcelain", "--untracked-files=no"))
    smpl = paths.find_smpl_model(workdir_root)
    return {
        "ready": bool(account["ok"] and code["ok"]),
        "cli": {"installed": version is not None, "version": version},
        "account": account,
        "gpu": gpu,
        "smpl": {"found": smpl is not None, "path": str(smpl) if smpl else None},
        "code": code,
    }
