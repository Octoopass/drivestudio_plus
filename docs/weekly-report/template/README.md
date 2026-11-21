# Mẫu báo cáo tuần

Mọi báo cáo tuần dùng cùng tên file, cùng bố cục, cùng cách viết và cùng một bảng công việc cộng dồn.
Người đọc chính là mentor, tức người ngoài nhóm: báo cáo phải tự đọc hiểu được mà không cần hỏi lại.

## Thư mục

```
docs/weekly-report/
├── tracker.yaml                 bảng công việc dùng chung (bản gốc duy nhất)
├── tuan-01_2026-11-07.md        báo cáo từng tuần, tên: tuan-NN_<ngày cuối tuần>.md
├── tuan-02_2026-11-14.md
├── ...
├── figures/tuan-NN/             hình của từng tuần
├── archive/                     bản gốc các báo cáo cũ (slide, poster, PDF) trước khi chuẩn hoá
├── build/                       PDF xuất ra (có đưa lên git)
└── template/
    ├── tuan-NN_YYYY-MM-DD.md    mẫu báo cáo
    ├── thuat-ngu.md             bảng thuật ngữ dùng chung
    ├── build.py                 xuất PDF
    └── style.css
```

Tuần tính từ Chủ nhật đến thứ Bảy; tên file lấy ngày thứ Bảy. Tên không dấu, không khoảng trắng để
link trên SharePoint và MS Teams không bị lỗi.

## Bố cục (giữ đúng thứ tự)

Thứ tự đi theo mạch: đứng ở đâu → đã hứa gì và làm được gì → làm cụ thể thế nào → đo được gì → vì
vậy quyết định gì → tuần sau làm gì → còn vướng gì.

| Mục | Nội dung | Cách có |
|---|---|---|
| Quy ước thuật ngữ | Bảng thuật ngữ dùng chung (method, cấu hình, renderer, track benchmark…), lấy từ `template/thuat-ngu.md` | Bảng tự sinh |
| 1. Tóm tắt | Dòng tiến độ, một câu trọng tâm của tuần, 3–5 ý chính kèm số, việc tuần sau | Dòng tiến độ tự sinh; phần còn lại viết tay |
| 2. Vị trí trên lộ trình | Một đến hai câu về mốc hiện tại, rồi bảng mốc M0–M5 và track A/B/C | Bảng tự sinh |
| 3. Đối chiếu với kế hoạch tuần trước | Từng việc trong kế hoạch tuần trước kèm trạng thái và kết quả thực tế; rồi việc tiếp tục từ trước; rồi việc ngoài kế hoạch (thêm mới trong quá trình làm) | Bảng tự sinh |
| 4. Chi tiết công việc đã làm | Tool, dữ liệu, khảo sát, thiết kế: mỗi việc một mục con | Viết tay |
| 5. Kết quả thực nghiệm | Số đo thật, mỗi thực nghiệm một mục con | Viết tay |
| 6. Phát hiện và quyết định | Quyết định rút ra từ mục 4 và 5, kèm lý do và nguồn | Viết tay |
| 7. Kế hoạch tuần sau | Bảng công việc và kết quả cần có; bảng chi tiết (như ma trận chạy) đặt ngay sau | Bảng kế hoạch tự sinh từ `ke_hoach` |
| 8. Hạn chế, rủi ro và câu hỏi còn mở | Điều kết quả chưa nói được, điều có thể làm trễ kế hoạch, câu hỏi cần mentor hoặc nhóm trả lời | Viết tay |
| 9. Bảng tổng hợp công việc | Cộng dồn từ tuần 01, chia khối theo tuần, khối cuối (xám) là kế hoạch tuần sau | Bảng tự sinh |

Mục nào không có nội dung thì giữ tiêu đề và ghi một câu nói rõ vì sao, ví dụ "Tuần này chưa có
thực nghiệm, vì nhóm dành thời gian cho thiết kế ở mục 4."

## Cách viết

- **Câu dẫn ở đầu mỗi mục và mục con.** Câu đầu cho biết phần này nói gì và nối với phần trước thế
  nào. Không mở đầu thẳng bằng danh sách hay bảng. Ví dụ, thay vì "drive2sim cho phép chạy ... qua sáu
  bước:", viết "Nhóm đã xây dựng phiên bản đầu tiên của tool drive2sim, cho phép chạy và so sánh các
  approach trên cùng dữ liệu, cùng split và cùng các cách đo. Tool đi qua sáu bước tổng thể:".
- **Câu đầy đủ.** Mỗi câu có chủ ngữ rõ (nhóm, tool, method nào); gạch đầu dòng cũng là câu đủ ý,
  không phải cụm từ rời. Thuật ngữ lạ giải thích ngắn ở lần đầu xuất hiện.
- **Bảng số liệu ghi đủ điều kiện chạy:** số bước, GPU, codebase (tool drive2sim hay repo gốc), dataset
  và đoạn khung, dưới dạng cột hoặc trong câu thiết lập. Thông tin không có thì ghi "chưa ghi lại",
  không bỏ trống. Run khác nhau nhiều thiết lập cùng lúc thì nói rõ là chưa tách được nguyên nhân.
- **Nhận xét phải dẫn tới kết luận.** Khi tuần đó có thực nghiệm so cấu hình, mục 6 có thêm bảng
  "Cấu hình khuyến nghị theo trường hợp" (Trường hợp · Cấu hình · Vì sao · Đánh đổi · Nguồn), không
  dừng ở các gạch đầu dòng nhận xét.
- **Kết luận trước, số liệu sau.** Mục con thực nghiệm mở đầu bằng kết luận kèm con số, rồi tới
  thiết lập, bảng, và "Từ bảng trên, nhóm rút ra các nhận xét sau:".
- **Không ghi gì dành cho người trong nhóm:** mã nội bộ (`id` trong `tracker.yaml`), đường dẫn file
  nội bộ, "Bản gốc: …", "cần bổ sung …". Việc còn thiếu thì sửa trước khi gửi.
- **Dùng đúng thuật ngữ trong `template/thuat-ngu.md`.** Method (approach) là OmniRe, Street GS…;
  cấu hình là method cộng lựa chọn bên trong (OmniRe + 3DGUT + FastGS + Faster-GS). Ghi rõ là dựng
  từng cảnh (per-scene optimization) hay dựng tức thì (feed-forward). Mỗi lần nhắc track benchmark thì
  ghi kèm tên: "track A (dựng từng cảnh)", "track B (quãng đường dài)", "track C (feed-forward)".
  Không gọi loại camera (pinhole, fisheye) là track; ID của box gọi là "ID theo dõi". Thêm thuật ngữ
  mới thì thêm vào `thuat-ngu.md` để mọi tuần dùng chung.
- **Tham chiếu chéo ghi cả số và tên mục:** "mục 5.2 (Fisheye trên KITTI-360)", không chỉ "mục 5.2"
  hay "bảng trên".

## Bảng và hình

- **Ba bảng chính mọi tuần đều có:** bảng lộ trình (mục 2), bảng quyết định (mục 6, ba cột: Quyết
  định · Lý do / bằng chứng · Nguồn), bảng tổng hợp công việc (mục 9). Cộng thêm bảng đối chiếu (mục 3)
  và bảng kế hoạch (mục 7), cả hai tự sinh.
- **Mỗi bảng chính có một dòng ghi chú in nghiêng ngay trên bảng**, nói bảng thể hiện gì. Bảng tự sinh
  có sẵn ghi chú; bảng viết tay thì thêm `*Ghi chú: …*`. Bảng phụ (số liệu thực nghiệm) cần một câu dẫn.
- **Cột "Nguồn"** ghi mục trong báo cáo (số và tên), paper (tên, hội nghị, năm) hoặc tài liệu có link.
  Mọi dataset, method, paper mới nhắc tới lần đầu phải có nguồn.
- **Hình** đặt trong `figures/tuan-NN/`, chèn bằng `![Hình N. …](figures/tuan-NN/ten.png)` rồi dòng
  `*Hình N. …*` ngay dưới; script giữ hình và chú thích trên cùng một trang. Không đặt hình giữa một
  danh sách. Nhiều ảnh nhỏ cùng hàng: `<div class="gallery"><img src="…"> …</div>`.
- Bảng từ 9 cột trở lên tự thu nhỏ chữ để vừa trang A4.
- **Hình so sánh kết quả:** mỗi thực nghiệm nên có ít nhất một hình đặt ảnh render cạnh ảnh thật ở
  cùng một khung test, và cạnh nhau giữa các cấu hình đang so. `template/compare_figure.py` cắt đúng
  camera ở cùng một khung từ video tập test của từng run rồi ghép thành lưới có nhãn:

  ```bash
  python template/compare_figure.py figures/tuan-NN/ten.jpg --frame 10 --camera CAM_FRONT --cols 2 \
      --panel "gt:8d9c:Ảnh thật" --panel "8d9c:OmniRe + 3DGS (25,20 dB)" --panel "f8a6:… (25,01 dB)"
  ```

  `<run>` là 4 ký tự cuối của run id; `--step` cho video khác bước 30.000; `--layout-run` mượn bố cục
  của run cùng split khi run thiếu file bố cục. Dưới hình viết một câu nói điều nhìn thấy trên ảnh mà
  bảng số không cho thấy.

## Màu trong bảng công việc

| Màu | `trang_thai` | Nghĩa |
|---|---|---|
| Xanh lá | `xong` | Đã xong trong tuần |
| Vàng | `dang-lam` | Đang làm, chuyển sang tuần sau |
| Đỏ | `dung` | Bị chặn hoặc bỏ (ghi lý do trong mục 6) |
| Xám | `ke-hoach` | Kế hoạch tuần sau |

Trong bảng đối chiếu (mục 3), việc có trong kế hoạch mà tuần đó không báo cáo hiện màu đỏ với nhãn
"Chưa báo cáo".

## Mỗi tuần làm gì

1. Trong `tracker.yaml`, ở tuần NN: chép các việc của `ke_hoach` và các việc tuần trước còn
   `dang-lam` vào `cong_viec`, đặt trạng thái thật, và ghi `ghi_chu` ngắn (kết quả thực tế, lý do
   dừng). Thêm việc phát sinh. Thêm tuần NN+1 chỉ có `ke_hoach`, mỗi việc kèm `ket_qua` (kết quả cần
   có để tính là xong). `viec` ghi tổng quát; chi tiết để ở mục 4 và 5 của báo cáo.
2. Chép `template/tuan-NN_YYYY-MM-DD.md` thành `tuan-NN_<ngày>.md`, viết các mục 1, 2 (câu dẫn), 4–8.
   Giữ nguyên sáu chỗ `{{thuat-ngu}}`, `{{tien-do}}`, `{{lo-trinh}}`, `{{thay-doi}}`, `{{ke-hoach}}`,
   `{{bang-cong-viec}}`.
3. Xuất PDF, đọc lại các cảnh báo in ra terminal (việc bị bỏ sót) và đọc lại PDF:

   ```bash
   cd docs/weekly-report
   ~/miniconda3/envs/drive2sim/bin/python template/build.py tuan-NN_<ngày>.md
   ```

4. Đưa PDF trong `build/` lên SharePoint và báo trên MS Teams kèm link.
