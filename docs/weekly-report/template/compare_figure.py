r"""Ghép hình so sánh cho báo cáo tuần: cùng một khung test, cùng một camera, từ nhiều run.

Mỗi run lưu video tập test ở ``d2s/stages/train.drivestudio/videos/``: ``test_set_<step>_rgbs.mp4``
(ảnh render) và ``test_set_<step>_gt_rgbs.mp4`` (ảnh thật), kèm ``test_set_<step>_layout.json``
ghi vị trí từng camera trong khung. Các run cùng cảnh, cùng split có cùng các khung test, nên lấy
khung thứ ``n`` của mỗi video là cùng một thời điểm. Script cắt đúng camera đó và ghép thành
lưới, mỗi ô có nhãn.

Ví dụ (chạy trong ``docs/weekly-report``)::

    python template/compare_figure.py figures/tuan-03/so-sanh.jpg --frame 10 --camera CAM_FRONT \
        --cols 2 --panel gt:8d9c:"Ảnh thật" --panel 8d9c:"OmniRe + 3DGS"

``--panel <run>:<nhãn>`` lấy ảnh render của run; ``--panel gt:<run>:<nhãn>`` lấy ảnh thật từ video
của run đó. ``<run>`` là 4 ký tự cuối của run id.
"""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RUNS = Path(__file__).resolve().parents[3] / "runs"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
STEP = 30000
LABEL_H = 34
GAP = 6


def find_run(short: str) -> Path:
    """Thư mục run có id kết thúc bằng ``short``."""
    found = sorted(RUNS.glob(f"*/*/*-{short}"))
    if len(found) != 1:
        raise SystemExit(f"không tìm được đúng một run '{short}': {found}")
    return found[0]


def frame(video: Path, index: int) -> Image.Image:
    """Khung thứ ``index`` (đếm từ 0) của video, dạng ảnh RGB."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{index})",
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        check=True, capture_output=True,
    ).stdout
    from io import BytesIO

    return Image.open(BytesIO(raw)).convert("RGB")


def layout_of(run: Path, step: int) -> Path:
    """File bố cục khung của ``run`` (tập test, rồi tập đầy đủ: cùng vị trí camera)."""
    videos = run / "d2s" / "stages" / "train.drivestudio" / "videos"
    for name in (f"test_set_{step}_layout.json", f"full_set_{step}_layout.json"):
        if (videos / name).is_file():
            return videos / name
    raise SystemExit(f"{run.name} không có file bố cục; dùng --layout-run")


def crop(
    run: Path, gt: bool, index: int, camera: str, step: int, layout_path: Path
) -> Image.Image:
    """Cắt ``camera`` ở khung ``index`` từ video render (hoặc ảnh thật) của ``run``."""
    videos = run / "d2s" / "stages" / "train.drivestudio" / "videos"
    layout = json.loads(layout_path.read_text())
    image = frame(videos / f"test_set_{step}_{'gt_rgbs' if gt else 'rgbs'}.mp4", index)
    # video bị đổi kích thước thành bội số của 16 khi ghi: đổi toạ độ theo tỉ lệ
    sx, sy = image.width / layout["width"], image.height / layout["height"]
    box = next(c for c in layout["cameras"] if c["name"] == camera)
    return image.crop((
        round(box["x"] * sx), round(box["y"] * sy),
        round((box["x"] + box["width"]) * sx), round((box["y"] + box["height"]) * sy),
    ))


def main() -> None:
    """Ghép lưới hình từ các ``--panel`` và lưu ra file JPEG."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--width", type=int, default=800, help="bề rộng mỗi ô (px)")
    parser.add_argument("--panel", action="append", required=True)
    parser.add_argument("--step", type=int, default=STEP, help="bước train của video")
    parser.add_argument(
        "--layout-run", help="mượn bố cục của run này cho run thiếu file bố cục (cùng split)"
    )
    args = parser.parse_args()
    font = ImageFont.truetype(FONT, 20)
    panels = []
    for spec in args.panel:
        gt = spec.startswith("gt:")
        short, label = (spec[3:] if gt else spec).split(":", 1)  # nhãn được chứa dấu ":"
        run = find_run(short)
        try:
            layout_path = layout_of(run, args.step)
        except SystemExit:
            if not args.layout_run:
                raise
            layout_path = layout_of(find_run(args.layout_run), args.step)
        image = crop(run, gt, args.frame, args.camera, args.step, layout_path)
        image = image.resize((args.width, round(image.height * args.width / image.width)))
        panel = Image.new("RGB", (image.width, image.height + LABEL_H), "white")
        panel.paste(image, (0, LABEL_H))
        ImageDraw.Draw(panel).text((6, 6), label, fill="black", font=font)
        panels.append(panel)
    rows = -(-len(panels) // args.cols)
    cell_w = max(p.width for p in panels)
    cell_h = max(p.height for p in panels)
    sheet = Image.new(
        "RGB", (args.cols * cell_w + (args.cols - 1) * GAP, rows * cell_h + (rows - 1) * GAP), "white"
    )
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % args.cols) * (cell_w + GAP), (i // args.cols) * (cell_h + GAP)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=88)
    print(args.output, sheet.size)


if __name__ == "__main__":
    main()
