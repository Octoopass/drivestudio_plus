# Real2Sim: báo cáo tiến độ — bộ nhớ của deck

Đọc file này trước khi sửa deck, và cập nhật nó: mỗi khi người dùng góp ý
hoặc sửa điều gì, ghi vào mục "Quyết định & góp ý" để không lặp lại lỗi cũ.

## Giới thiệu
- Dịp / mục tiêu: <ĐIỀN VÀO>
- Khán giả: <ĐIỀN VÀO — là ai, đã biết gì>
- Thời lượng: <ĐIỀN VÀO> phút · deck để đọc hay để nói: <ĐIỀN VÀO>
- Ngôn ngữ & giọng văn: <ĐIỀN VÀO — ví dụ: dễ hiểu, gần gũi, ít công thức>

## Thiết kế
- Mẫu: `blue-professional` — giữ nguyên font, bảng màu, lưới bố cục và chi tiết trang trí
  (deck-kit AGENTS.md §3; mô tả đầy đủ ở templates/blue-professional/design.md)
- Ý nghĩa màu: <ĐIỀN VÀO — ví dụ: xanh dương = dự đoán, cam = phép đo>
- Quy tắc: <ĐIỀN VÀO — ví dụ: tối đa 4 gạch đầu dòng mỗi slide>

## Nội dung
- Nguồn / số liệu phải theo: <ĐIỀN VÀO>
- Từ ngữ hoặc khẳng định cần tránh: <ĐIỀN VÀO>

## Quyết định & góp ý (mới nhất ở trên)
- 2026-11-07 — tạo deck.

## Cách làm việc với deck này
- `index.html` là deck. Mỗi slide có `data-title`, `data-sec`, `data-min`, ghi chú người nói trong `<aside class="notes">`.
- Người dùng có thể chỉ một phần tử bằng phím **K** (chép đường dẫn file, dòng, slide và selector) — sửa đúng phần tử đó.
- Sửa xong chạy check-deck.mjs; chia sẻ bằng export.mjs hoặc deploy.mjs (deck-kit/scripts).

- 2026-11-07: Sơ đồ lấy từ report (TikZ xuất SVG) trong figures/. Khi report đổi sơ đồ thì phải xuất lại.
