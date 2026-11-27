# Báo cáo tiến độ tuần NN · DD/MM/YYYY

Reconstructing Real-World Driving Scenes (Real2Sim) · Tuần NN: DD/MM/YYYY – DD/MM/YYYY

<!--
Mẫu báo cáo tuần. Quy tắc đầy đủ trong template/README.md, tóm tắt:
- Giữ đúng 9 mục và thứ tự dưới đây; mục nào không có nội dung thì ghi một câu nói rõ vì sao.
- Mỗi mục và mỗi mục con mở đầu bằng một câu dẫn: phần này nói gì và nối với phần trước thế nào.
  Ví dụ: "Nhóm đã xây dựng phiên bản đầu tiên của tool drive2sim, cho phép chạy và so sánh các
  approach trên cùng dữ liệu, ... Tool đi qua sáu bước tổng thể:". Không mở đầu thẳng bằng danh
  sách hoặc bảng.
- Viết câu đầy đủ chủ ngữ, vị ngữ; giải thích thuật ngữ lạ ở lần đầu xuất hiện.
- Dùng đúng thuật ngữ trong template/thuat-ngu.md: method khác cấu hình; ghi rõ dựng từng cảnh
  (per-scene optimization) hay feed-forward; track benchmark luôn kèm tên, ví dụ "track B (quãng
  đường dài)"; pinhole/fisheye là loại camera, không phải track.
- Mỗi bảng viết tay có một câu dẫn trước bảng; bảng chính có thêm một dòng *Ghi chú: …*.
- Khi nhắc tới phần khác, ghi cả số mục và tên mục: "mục 5.2 (Fisheye trên KITTI-360)".
- Hình: ![Hình N. …](figures/tuan-NN/ten.png), rồi dòng *Hình N. …* ngay dưới.
- Không ghi mã nội bộ (id trong tracker.yaml), đường dẫn file nội bộ, hay ghi chú cho người trong
  nhóm (kiểu "Bản gốc: ...", "cần bổ sung"): báo cáo dành cho người ngoài nhóm đọc.
-->

## Quy ước thuật ngữ

{{thuat-ngu}}

## 1. Tóm tắt

{{tien-do}}

Một câu nói tuần này nhóm tập trung vào việc gì. Sau đó 3–5 gạch đầu dòng, mỗi dòng là một câu đầy đủ nêu một kết quả kèm số đo hoặc một quyết định. Dòng cuối nói tuần sau làm gì.

- **…:** …
- **Tuần sau:** …

## 2. Vị trí trên lộ trình

Một đến hai câu: dự án đang ở mốc nào, mốc nào đã xong và vì sao coi là xong.

{{lo-trinh}}

## 3. Đối chiếu với kế hoạch tuần trước

Một câu nhắc lại tuần trước đặt ra những việc gì, rồi dẫn vào bảng đối chiếu bên dưới.

{{thay-doi}}

## 4. Chi tiết công việc đã làm

Một câu nối các việc trong tuần với nhau (tool, dữ liệu, khảo sát, thiết kế) và cho biết chúng phục vụ thực nghiệm ở mục 5 thế nào.

### 4.1 Tên công việc (giống tên trong bảng công việc)

Câu dẫn: làm việc này để giải quyết điều gì. Tiếp theo là đã làm gì và kết quả ra sao. Có nhiều mục song song thì dùng danh sách hoặc bảng, và có câu dẫn trước danh sách hoặc bảng đó.

## 5. Kết quả thực nghiệm

Một câu nói tuần này chạy những nhóm thực nghiệm nào và dựa trên phần nào ở mục 4. Nếu chưa có thực nghiệm, ghi rõ vì sao và khi nào sẽ có.

### 5.1 Tên thực nghiệm

Câu đầu là kết luận kèm con số. Câu tiếp theo là thiết lập: dataset, cảnh, split, số bước, GPU, codebase. Thông tin không có thì ghi "chưa ghi lại".

| Run | Cấu hình | PSNR | SSIM | LPIPS | Train | VRAM |
|---|---|---|---|---|---|---|

Từ bảng trên, nhóm rút ra các nhận xét sau:

- …

## 6. Phát hiện và quyết định

Một câu dẫn: các quyết định dưới đây rút ra từ mục 4 và mục 5.

Nếu tuần này có thực nghiệm so cấu hình, gom các nhận xét ở mục 5 thành kết luận: nên dùng cấu hình nào cho trường hợp nào.

*Ghi chú: mỗi dòng là một trường hợp sử dụng; cột "Đánh đổi" ghi cái giá phải trả so với cấu hình khác.*

| Trường hợp | Cấu hình khuyến nghị | Vì sao | Đánh đổi | Nguồn |
|---|---|---|---|---|
| … | … | … | … | Mục 5.1 (Tên thực nghiệm) |

*Ghi chú: các quyết định chốt trong tuần, mỗi quyết định kèm lý do và nguồn (mục trong báo cáo, paper hoặc tài liệu).*

| Quyết định | Lý do / bằng chứng | Nguồn |
|---|---|---|
| … | … | Mục 5.1 (Tên thực nghiệm) |

## 7. Kế hoạch tuần sau

Một câu nói kế hoạch tuần sau xuất phát từ kết quả và quyết định nào ở trên. Bảng chi tiết (ví dụ ma trận chạy) đặt ngay sau bảng kế hoạch, có câu dẫn và dòng ghi chú.

{{ke-hoach}}

## 8. Hạn chế, rủi ro và câu hỏi còn mở

Một câu dẫn, rồi danh sách đánh số: điều kết quả chưa nói được, điều có thể làm trễ kế hoạch, câu hỏi cần mentor hoặc nhóm trả lời.

1. …

## 9. Bảng tổng hợp công việc

{{bang-cong-viec}}
