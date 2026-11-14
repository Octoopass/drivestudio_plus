"""Vòng hỏi nền dùng chung cho việc theo dõi run chạy ở máy khác (Kaggle, Colab).

``KaggleWatcher`` (``kaggle_watch.py``) và ``ColabWatcher`` (``colab_watch.py``) chỉ khác ở một
lượt hỏi (``check_once``): hỏi trạng thái gì, xong thì xếp việc tải nào. Phần còn lại giống nhau
và nằm ở đây: luồng nền, cứ ``interval_s`` giây hỏi một lượt; lượt hỏi lỗi (mạng, chưa đăng nhập,
bị giới hạn tần suất) thì khoảng chờ nhân đôi, tối đa ``max_interval_s``; hỏi được thì về lại
``interval_s``.

File này chứa ``RemoteWatcher``.
"""

import logging
import threading
from typing import ClassVar, Optional

LOGGER = logging.getLogger(__name__)


class RemoteWatcher:
    """Luồng nền gọi ``check_once`` định kỳ, giãn khoảng chờ khi lỗi.

    Lớp con đặt ``THREAD_NAME``, ``SERVICE`` (tên hiện trong log) và cài ``check_once``.

    Args:
        interval_s: Khoảng giữa hai lượt hỏi khi mọi thứ bình thường.
        max_interval_s: Khoảng chờ tối đa khi hỏi lỗi liên tiếp.
    """

    THREAD_NAME: ClassVar[str] = "remote-watch"
    SERVICE: ClassVar[str] = "máy chạy"

    def __init__(self, interval_s: float, max_interval_s: float) -> None:
        self.interval_s = interval_s
        self.max_interval_s = max(max_interval_s, interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check_once(self) -> list[str]:
        """Hỏi một lượt; trả ``run_id`` vừa được xếp việc tải. Lớp con cài."""
        raise NotImplementedError

    def _loop(self) -> None:
        delay = self.interval_s
        while not self._stop.wait(delay):
            try:
                self.check_once()
                delay = self.interval_s
            except Exception as error:  # noqa: BLE001  (luồng nền không được chết vì lỗi mạng)
                delay = min(delay * 2, self.max_interval_s)
                LOGGER.warning(
                    "không hỏi được %s (%s); thử lại sau %.0f giây", self.SERVICE, error, delay
                )

    def start(self) -> None:
        """Bật luồng nền; lượt hỏi đầu tiên sau ``interval_s``."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name=self.THREAD_NAME, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Dừng luồng nền (không chờ lượt hỏi đang chạy)."""
        self._stop.set()
