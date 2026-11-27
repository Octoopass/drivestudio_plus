r"""Xuất báo cáo tuần Markdown ra PDF, chèn bảng công việc có màu lấy từ ``tracker.yaml``.

Mỗi báo cáo tuần ``tuan-NN_YYYY-MM-DD.md`` viết bằng Markdown. Sáu chỗ trong file được thay bằng
nội dung sinh từ ``../tracker.yaml`` (bản gốc duy nhất của bảng công việc, mọi tuần dùng chung):

- ``{{tien-do}}``: một dòng đếm việc đã xong / đang làm / dừng trên tổng số việc tới tuần NN.
- ``{{thuat-ngu}}``: bảng thuật ngữ dùng chung (``template/thuat-ngu.md``).
- ``{{lo-trinh}}``: bảng mốc M0–M5 và track benchmark A/B/C, ô tô màu theo trạng thái của từng tuần.
- ``{{thay-doi}}``: bảng đối chiếu kế hoạch tuần trước với kết quả tuần này, cộng việc tiếp tục
  và việc thêm mới; việc trong kế hoạch mà không báo cáo ghi "Chưa báo cáo" (in thêm cảnh báo).
- ``{{ke-hoach}}``: bảng kế hoạch tuần NN+1 (công việc, kết quả cần có).
- ``{{bang-cong-viec}}``: bảng tổng hợp, chia khối theo tuần từ tuần 01 tới tuần NN, cộng khối
  kế hoạch tuần NN+1 tô xám.

Màu: xanh lá = xong, vàng = đang làm, đỏ = dừng (bị chặn hoặc bỏ), xám = kế hoạch.

Cần Python có ``markdown`` và ``pyyaml`` (môi trường ``drive2sim`` có sẵn) và Google Chrome::

    python template/build.py tuan-03_2026-11-21.md        # chạy trong docs/weekly-report/
    python template/build.py --all                        # xuất mọi tuần

PDF ghi vào ``build/`` cạnh file Markdown.
"""

import argparse
import datetime as dt
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRACKER = ROOT / "tracker.yaml"
CSS = HERE / "style.css"
GLOSSARY = HERE / "thuat-ngu.md"
BUILD = ROOT / "build"
REPORT_NAME = re.compile(r"^tuan-(\d{2})_\d{4}-\d{2}-\d{2}\.md$")

STATUS = {
    "xong": ("Xong", "st-xong"),
    "dang-lam": ("Đang làm", "st-dang-lam"),
    "dung": ("Dừng", "st-dung"),
    "ke-hoach": ("Kế hoạch", "st-ke-hoach"),
}
OPEN = {"dang-lam", "ke-hoach"}

# Dòng ghi chú đặt trên mỗi bảng tự sinh.
CAPTION_ROADMAP = (
    "Ghi chú: mốc M0–M5 và ba track benchmark A/B/C, mỗi cột là một tuần, ô tô màu theo trạng thái. "
    "Cam kết M0–M2 trong 6 tuần, M3–M5 làm theo mức khả thi nhất có thể. Trạng thái từng việc "
    "ở mục 9 (Bảng tổng hợp công việc)."
)
CAPTION_PLAN = (
    "Ghi chú: việc dự kiến làm trong tuần sau và kết quả cần có để tính là xong. Báo cáo tuần sau "
    "báo lại trạng thái của từng việc này ở mục 3 (Đối chiếu với kế hoạch tuần trước)."
)
CAPTION_CHANGES = (
    "Ghi chú: đối chiếu với bảng Kế hoạch tuần sau ở mục 7 của báo cáo tuần trước: việc đã đặt "
    "ra, trạng thái cuối tuần này và kết quả thực tế; cuối bảng là việc tiếp tục và việc thêm mới. "
    "Chi tiết ở mục 4 (Chi tiết công việc đã làm) và mục 5 (Kết quả thực nghiệm)."
)
CAPTION_GLOSSARY = (
    "Ghi chú: cách dùng các thuật ngữ trong báo cáo này; mọi báo cáo tuần dùng chung một bảng."
)
CAPTION_TRACKER = (
    "Ghi chú: cộng dồn từ tuần 01, mỗi khối là một tuần; việc kéo dài nhiều tuần lặp lại ở từng "
    "tuần để thấy màu đổi dần. Khối cuối (xám) là kế hoạch tuần sau, chi tiết ở mục 7 (Kế hoạch "
    "tuần sau)."
)


def caption(text: str) -> str:
    """Dòng ghi chú in nghiêng, đặt ngay trên bảng."""
    return f'<p class="caption">{html.escape(text)}</p>'


def load_tracker() -> dict:
    """Đọc ``tracker.yaml`` và kiểm tra trạng thái, ID, thứ tự tuần."""
    data = yaml.safe_load(TRACKER.read_text(encoding="utf-8"))
    weeks = data["tuan"]
    for i, week in enumerate(weeks, start=1):
        if week["so"] != i:
            sys.exit(f"tracker.yaml: tuần thứ {i} ghi so={week['so']}")
        for key in ("cong_viec", "ke_hoach"):
            seen = set()
            for item in week.get(key) or []:
                if item["id"] in seen:
                    sys.exit(f"tracker.yaml: tuần {i} có hai dòng {item['id']} trong {key}")
                seen.add(item["id"])
                status = item.get("trang_thai", "ke-hoach")
                if status not in STATUS:
                    sys.exit(f"tracker.yaml: {item['id']} trạng thái lạ '{status}'")
    return data


def fmt_date(value: dt.date) -> str:
    """``2026-11-07`` thành ``07/11/2026``."""
    return value.strftime("%d/%m/%Y")


def week_label(week: dict, plan: bool = False) -> str:
    """Tiêu đề khối tuần: ``Tuần 03 · 15/11/2026 – 21/11/2026``."""
    label = f"Tuần {week['so']:02d} · {fmt_date(week['bat_dau'])} – {fmt_date(week['ket_thuc'])}"
    return label + " (kế hoạch)" if plan else label


def latest_status(weeks: list[dict], upto: int) -> dict[str, str]:
    """Trạng thái mới nhất của từng ID tính tới hết tuần ``upto``."""
    latest: dict[str, str] = {}
    for week in weeks[:upto]:
        for item in week.get("cong_viec") or []:
            latest[item["id"]] = item["trang_thai"]
    return latest


def render_progress(weeks: list[dict], n: int) -> str:
    """Dòng tiến độ: số việc theo trạng thái mới nhất, kèm số việc mới xong trong tuần."""
    latest = latest_status(weeks, n)
    before = latest_status(weeks, n - 1)
    counts = {k: sum(1 for s in latest.values() if s == k) for k in STATUS}
    newly_done = sum(1 for i, s in latest.items() if s == "xong" and before.get(i) != "xong")
    plan = len(weeks[n].get("ke_hoach") or []) if n < len(weeks) else 0
    parts = [
        f'<span class="chip st-xong">Xong {counts["xong"]}/{len(latest)} (+{newly_done} tuần này)</span>',
        f'<span class="chip st-dang-lam">Đang làm {counts["dang-lam"]}</span>',
        f'<span class="chip st-dung">Dừng {counts["dung"]}</span>',
        f'<span class="chip st-ke-hoach">Kế hoạch tuần sau {plan}</span>',
    ]
    return '<p class="progress">' + " ".join(parts) + "</p>"


def render_roadmap(data: dict, n: int) -> str:
    """Bảng mốc và track: mỗi cột một tuần, ô tô màu theo trạng thái tuần đó."""
    weeks = data["tuan"][:n]
    head = "".join(f"<th>T{w['so']:02d}</th>" for w in weeks)
    rows = []
    for group in data["lo_trinh"]:
        rows.append(f'<tr class="group"><td colspan="{len(weeks) + 1}">{html.escape(group["nhom"])}</td></tr>')
        for item in group["muc"]:
            cells = []
            for week in weeks:
                status = (week.get("lo_trinh") or {}).get(item["id"], "ke-hoach")
                cells.append(f'<td class="{STATUS[status][1]}">{STATUS[status][0]}</td>')
            name = f"<b>{html.escape(item['id'])}</b> {html.escape(item['ten'])}"
            rows.append(f"<tr><td>{name}</td>{''.join(cells)}</tr>")
    return (
        caption(CAPTION_ROADMAP)
        + '<table class="roadmap"><thead><tr><th>Mốc / track benchmark</th>'
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_changes(weeks: list[dict], n: int) -> str:
    """Đối chiếu kế hoạch tuần trước với kết quả tuần này, thành một bảng ba khối.

    Khối 1: từng việc trong bảng "Kế hoạch tuần sau" của báo cáo tuần trước (``ke_hoach`` của tuần
    NN), kèm trạng thái và kết quả thực tế (``ghi_chu``) tuần này. Việc trong kế hoạch mà tuần này
    không có dòng nào thì ghi "Chưa báo cáo" và in cảnh báo. Khối 2: việc tuần trước còn đang làm,
    không có trong kế hoạch. Khối 3: việc thêm mới trong quá trình làm.
    """
    week = weeks[n - 1]
    items = {i["id"]: i for i in week.get("cong_viec") or []}
    prev = {i["id"]: i for i in (weeks[n - 2].get("cong_viec") or [])} if n > 1 else {}
    plan = week.get("ke_hoach") or []
    planned = {i["id"] for i in plan}
    with_result = any(i.get("ket_qua") for i in plan)
    cols = 4 if with_result else 3

    def row(name: str, wanted: str | None, status: str | None, actual: str) -> str:
        label, cls = ("Chưa báo cáo", "st-dung") if status is None else STATUS[status]
        cells = f"<td>{html.escape(name)}</td>"
        if with_result:
            cells += f"<td>{html.escape(wanted or '')}</td>"
        cells += f'<td><span class="chip {cls}">{label}</span></td><td>{html.escape(actual)}</td>'
        return f'<tr class="{cls}">{cells}</tr>'

    def group(label: str) -> str:
        return f'<tr class="group"><td colspan="{cols}">{html.escape(label)}</td></tr>'

    def warn(text: str) -> None:
        print(f"cảnh báo: tuần {n:02d} {text}", file=sys.stderr)

    rows = []
    if plan:
        rows.append(group("Theo kế hoạch tuần trước"))
        for p in plan:
            item = items.get(p["id"])
            if item is None:
                warn(f"không báo cáo việc trong kế hoạch: {p['viec']}")
                rows.append(row(p["viec"], p.get("ket_qua"), None, ""))
            else:
                rows.append(row(p["viec"], p.get("ket_qua"), item["trang_thai"], item.get("ghi_chu") or ""))
    carried = [i for i in prev.values() if i["trang_thai"] in OPEN and i["id"] not in planned]
    if carried:
        rows.append(group("Tiếp tục từ tuần trước (đang làm, không có trong kế hoạch)"))
        for p in carried:
            item = items.get(p["id"])
            if item is None:
                warn(f"không nhắc lại việc tuần trước còn mở: {p['viec']}")
                rows.append(row(p["viec"], None, None, ""))
            else:
                rows.append(row(item["viec"], None, item["trang_thai"], item.get("ghi_chu") or ""))
    extra = [i for i_id, i in items.items() if i_id not in planned and i_id not in prev]
    if extra:
        if n > 1:
            rows.append(group("Ngoài kế hoạch (thêm mới trong quá trình làm)"))
        else:
            rows.append(group("Tuần đầu tiên: chưa có kế hoạch từ tuần trước"))
        for item in extra:
            rows.append(row(item["viec"], None, item["trang_thai"], item.get("ghi_chu") or ""))
    head = "<th>Công việc</th>" + ("<th>Kết quả cần có</th>" if with_result else "")
    head += "<th>Trạng thái</th><th>Kết quả thực tế</th>"
    return (
        caption(CAPTION_CHANGES)
        + f'<table class="changes"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def render_tracker(weeks: list[dict], n: int) -> str:
    """Bảng tổng hợp: khối tuần 01..NN theo trạng thái thật, khối NN+1 là kế hoạch (xám)."""
    rows = []

    def block(label: str, items: list[dict], plan: bool) -> None:
        rows.append(f'<tr class="group"><td colspan="2">{html.escape(label)}</td></tr>')
        for item in items:
            status = "ke-hoach" if plan else item["trang_thai"]
            rows.append(
                f'<tr class="{STATUS[status][1]}">'
                f"<td>{html.escape(item['viec'])}</td>"
                f"<td>{html.escape(item.get('ghi_chu') or '')}</td></tr>"
            )

    for week in weeks[:n]:
        block(week_label(week), week.get("cong_viec") or [], plan=False)
    if n < len(weeks) and weeks[n].get("ke_hoach"):
        block(week_label(weeks[n], plan=True), weeks[n]["ke_hoach"], plan=True)
    legend = "".join(f'<span class="chip {c}">{t}</span> ' for t, c in STATUS.values())
    return (
        caption(CAPTION_TRACKER)
        + f'<p class="legend">{legend}</p>'
        '<table class="tracker"><thead><tr><th>Công việc</th><th>Ghi chú</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_plan(weeks: list[dict], n: int) -> str:
    """Bảng kế hoạch tuần NN+1 lấy từ ``ke_hoach``: công việc và kết quả cần có."""
    items = weeks[n].get("ke_hoach") or [] if n < len(weeks) else []
    if not items:
        return "<p>Chưa có kế hoạch tuần sau trong tracker.yaml.</p>"
    with_result = any(i.get("ket_qua") for i in items)
    head = "<th>Công việc</th>" + ("<th>Kết quả cần có</th>" if with_result else "")
    rows = []
    for item in items:
        cells = f"<td>{html.escape(item['viec'])}</td>"
        if with_result:
            cells += f"<td>{html.escape(item.get('ket_qua') or '')}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return (
        caption(CAPTION_PLAN)
        + f'<p><b>{html.escape(week_label(weeks[n]))}</b></p>'
        + f'<table class="plan"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def build(report: Path, data: dict) -> Path:
    """Ghép báo cáo, sinh HTML rồi in ra ``build/<tên>.pdf`` bằng Chrome headless."""
    match = REPORT_NAME.match(report.name)
    if not match:
        sys.exit(f"{report.name}: tên phải dạng tuan-NN_YYYY-MM-DD.md")
    n = int(match.group(1))
    weeks = data["tuan"]
    if n > len(weeks):
        sys.exit(f"tracker.yaml chưa có tuần {n}")
    text = report.read_text(encoding="utf-8")
    parts = {
        "{{thuat-ngu}}": caption(CAPTION_GLOSSARY)
        + markdown.markdown(GLOSSARY.read_text(encoding="utf-8"), extensions=["tables"]),
        "{{tien-do}}": render_progress(weeks, n),
        "{{lo-trinh}}": render_roadmap(data, n),
        "{{thay-doi}}": render_changes(weeks, n),
        "{{ke-hoach}}": render_plan(weeks, n),
        "{{bang-cong-viec}}": render_tracker(weeks, n),
    }
    slots = {}
    for key, value in parts.items():
        slot = f"D2SSLOT{len(slots)}"
        slots[slot] = value
        text = text.replace(key, f"\n\n{slot}\n\n")
    body = markdown.markdown(text, extensions=["tables", "sane_lists", "fenced_code"])
    for slot, value in slots.items():
        body = body.replace(f"<p>{slot}</p>", value)
    # hình (ảnh hoặc dãy ảnh) và dòng chú thích in nghiêng ngay sau thành một khối không tách trang
    body = re.sub(
        r'(<p><img [^>]*></p>|<div class="gallery">.*?</div>)\s*(<p><em>[^<]*</em></p>)',
        r'<figure>\1\2</figure>',
        body,
        flags=re.S,
    )
    title = re.search(r"^# (.+)$", text, re.M)
    page = (
        "<!doctype html><html lang=vi><head><meta charset=utf-8>"
        f'<base href="{report.parent.as_uri()}/">'
        f"<title>{html.escape(title.group(1) if title else report.stem)}</title>"
        f"<style>{CSS.read_text(encoding='utf-8')}</style></head><body>{body}</body></html>"
    )
    BUILD.mkdir(exist_ok=True)
    out = BUILD / f"{report.stem}.pdf"
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if chrome is None:
        sys.exit("không tìm thấy Google Chrome / Chromium")
    with tempfile.TemporaryDirectory() as tmp:
        page_file = Path(tmp) / "report.html"
        page_file.write_text(page, encoding="utf-8")
        (BUILD / f"{report.stem}.html").write_text(page, encoding="utf-8")
        subprocess.run(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={tmp}/profile",
                f"--print-to-pdf={out}",
                page_file.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    return out


def main() -> None:
    """Xuất các báo cáo trên dòng lệnh (hoặc mọi báo cáo với ``--all``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    reports = sorted(ROOT.glob("tuan-*.md")) if args.all else [r.resolve() for r in args.reports]
    if not reports:
        parser.error("cần tên file báo cáo hoặc --all")
    data = load_tracker()
    for report in reports:
        print(build(report, data))


if __name__ == "__main__":
    main()
