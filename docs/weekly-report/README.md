# Báo cáo tuần

Mỗi tuần một file `tuan-NN_<ngày thứ Bảy>.md`, cùng bố cục, cùng một bảng công việc cộng dồn trong
`tracker.yaml`. Mẫu báo cáo và các bước cập nhật mỗi tuần: [`template/README.md`](template/README.md).

## Cách dùng

Chạy `python template/build.py --all` bằng Python của môi trường `drive2sim` để xuất PDF:

```bash
cd docs/weekly-report
~/miniconda3/envs/drive2sim/bin/python template/build.py --all                   # mọi tuần
~/miniconda3/envs/drive2sim/bin/python template/build.py tuan-03_2026-11-21.md   # một tuần
```

PDF ghi vào `build/`. Script tự sinh 6 phần, từ `tracker.yaml` và `template/thuat-ngu.md`:

- bảng quy ước thuật ngữ ở đầu báo cáo;
- dòng tiến độ, ví dụ tuần 03: "Xong 13/21 (+6 tuần này)";
- bảng lộ trình M0–M5 và ba track benchmark A/B/C, tô màu theo từng tuần;
- bảng đối chiếu với kế hoạch tuần trước (đã đặt ra gì, làm được gì);
- bảng kế hoạch tuần sau, kèm kết quả cần có;
- bảng tổng hợp chia khối theo tuần, khối cuối màu xám là kế hoạch tuần sau.

Nếu một việc tuần trước còn mở mà tuần này không nhắc lại, script in cảnh báo và ghi thẳng vào PDF.

## Thư mục

| Đường dẫn | Nội dung |
|---|---|
| `tracker.yaml` | Bảng công việc chung, mọi tuần chỉ sửa file này |
| `tuan-NN_YYYY-MM-DD.md` | Báo cáo từng tuần |
| `figures/tuan-NN/` | Hình của từng tuần |
| `archive/` | Bản gốc các báo cáo cũ (poster, PDF) trước khi chuẩn hoá |
| `build/` | PDF (và HTML trung gian) xuất ra, có đưa lên git để gửi link |
| `template/` | Mẫu báo cáo, hướng dẫn, `build.py` (xuất PDF), `compare_figure.py` (ghép hình so sánh), `style.css` |
