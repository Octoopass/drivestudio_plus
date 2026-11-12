"""Trung chuyển sự kiện tiến độ từ engine tới trình duyệt qua Server-Sent Events (SSE).

Engine chạy run trong một luồng nền và phát sự kiện ``{"event": ..., "run_id": ..., ...}``
(CONTRACTS mục 11.4). Trình duyệt mở kết nối SSE để nhận các sự kiện đó gần như ngay lập tức.
File này nối hai đầu:
- ``EventBroker``: nhận sự kiện từ luồng nền (``publish``), đánh số tăng dần theo từng run, giữ
  1000 sự kiện gần nhất của mỗi run, và chuyển sự kiện tới mọi người đang theo dõi.
- ``format_sse``: viết một sự kiện thành đoạn văn bản theo định dạng SSE.
- ``stream_events``: hàm sinh dữ liệu cho một kết nối SSE: gửi trạng thái hiện tại trước, gửi lại
  sự kiện bị lỡ theo ``Last-Event-ID``, rồi gửi sự kiện mới, kèm dòng giữ kết nối mỗi 15 giây.

Routes trong ``drive2sim/studio/api/routes/runs.py`` dùng file này.
"""

import asyncio
import json
import threading
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional

HISTORY_SIZE = 1000
KEEPALIVE_S = 15.0

Event = dict[str, object]


@dataclass
class _Subscriber:
    """Một kết nối SSE đang theo dõi; ``run_id`` là ``None`` khi theo dõi mọi run."""

    run_id: Optional[str]
    queue: "asyncio.Queue[tuple[int, Event]]"
    loop: asyncio.AbstractEventLoop


class EventBroker:
    """Nhận sự kiện từ luồng chạy run và chuyển tới các kết nối SSE.

    An toàn khi ``publish`` được gọi từ luồng khác với vòng lặp asyncio của server: sự kiện được
    đưa vào hàng đợi của từng người theo dõi bằng ``call_soon_threadsafe``.
    """

    def __init__(self, history_size: int = HISTORY_SIZE) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, deque[tuple[int, Event]]] = {}
        self._next_id: dict[str, int] = {}
        self._history_size = history_size
        self._subscribers: list[_Subscriber] = []

    def publish(self, event: Event) -> None:
        """Ghi nhận một sự kiện và chuyển nó tới những ai đang theo dõi run đó hoặc mọi run.

        Args:
            event: Sự kiện của engine, có ``run_id``.
        """
        run_id = str(event.get("run_id", ""))
        with self._lock:
            event_id = self._next_id.get(run_id, 0) + 1
            self._next_id[run_id] = event_id
            history = self._history.setdefault(run_id, deque(maxlen=self._history_size))
            history.append((event_id, event))
            targets = [s for s in self._subscribers if s.run_id in (None, run_id)]
        for subscriber in targets:
            subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, (event_id, event))

    def subscribe(
        self, run_id: Optional[str], last_event_id: Optional[int] = None
    ) -> tuple[_Subscriber, list[tuple[int, Event]]]:
        """Đăng ký theo dõi một run (hoặc mọi run) từ vòng lặp asyncio hiện tại.

        Args:
            run_id: Run cần theo dõi; ``None`` để theo dõi mọi run.
            last_event_id: Số của sự kiện cuối client đã nhận (header ``Last-Event-ID``); các
                sự kiện sau số này còn trong bộ nhớ được gửi lại.

        Returns:
            Bộ ``(người theo dõi, các sự kiện bị lỡ)``. Gọi ``unsubscribe`` khi kết nối đóng.
        """
        subscriber = _Subscriber(run_id, asyncio.Queue(), asyncio.get_running_loop())
        with self._lock:
            self._subscribers.append(subscriber)
            missed: list[tuple[int, Event]] = []
            if run_id is not None and last_event_id is not None:
                missed = [
                    item for item in self._history.get(run_id, ()) if item[0] > last_event_id
                ]
        return subscriber, missed

    def unsubscribe(self, subscriber: _Subscriber) -> None:
        """Bỏ một người theo dõi khi kết nối SSE đóng."""
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)


def format_sse(event: Event, event_id: Optional[int] = None) -> str:
    """Viết một sự kiện theo định dạng SSE.

    Args:
        event: Sự kiện; trường ``event`` thành tên sự kiện SSE, phần còn lại thành ``data``.
        event_id: Số của sự kiện, ghi vào dòng ``id:`` để client gửi lại khi kết nối lại.

    Returns:
        Đoạn văn bản gồm dòng ``event: <tên>``, dòng ``id: <số>`` (nếu có), dòng
        ``data: <JSON>`` và một dòng trống.
    """
    data = {key: value for key, value in event.items() if key != "event"}
    lines = [f"event: {event.get('event', 'message')}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False, default=str))
    return "\n".join(lines) + "\n\n"


async def stream_events(
    broker: EventBroker,
    run_id: Optional[str],
    initial: list[Event],
    last_event_id: Optional[int] = None,
    keepalive_s: float = KEEPALIVE_S,
) -> AsyncIterator[str]:
    """Sinh dữ liệu cho một kết nối SSE.

    Gửi các sự kiện ``initial`` (trạng thái hiện tại) trước, để giao diện mở muộn vẫn đúng; rồi
    các sự kiện bị lỡ; rồi sự kiện mới. Không có sự kiện trong ``keepalive_s`` giây thì gửi dòng
    ``: keepalive`` để kết nối không bị proxy hoặc port forward cắt. Khi theo dõi một run, kết
    nối đóng sau sự kiện ``done`` (CONTRACTS mục 11.4).

    Args:
        broker: Nơi trung chuyển sự kiện.
        run_id: Run cần theo dõi, hoặc ``None`` cho mọi run.
        initial: Sự kiện gửi ngay khi kết nối, thường là trạng thái hiện tại.
        last_event_id: Header ``Last-Event-ID`` của client, nếu có.
        keepalive_s: Khoảng giữa hai dòng giữ kết nối, giây.

    Yields:
        Từng đoạn văn bản SSE.
    """
    subscriber, missed = broker.subscribe(run_id, last_event_id)
    try:
        for event in initial:
            yield format_sse(event)
            if run_id is not None and event.get("event") == "done":
                return
        for event_id, event in missed:
            yield format_sse(event, event_id)
            if run_id is not None and event.get("event") == "done":
                return
        while True:
            try:
                event_id, event = await asyncio.wait_for(subscriber.queue.get(), keepalive_s)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield format_sse(event, event_id)
            if run_id is not None and event.get("event") == "done":
                return
    finally:
        broker.unsubscribe(subscriber)
