# drive2sim: đặc tả yêu cầu

`drive2sim` là tool dựng lại cảnh lái xe thành cảnh 3D Gaussian Splatting (3DGS) để dùng cho mô phỏng. Tool được xây ngay trên repo này (fork của `drivestudio_plus`, vốn là fork của DriveStudio): phần tái dựng dùng lõi DriveStudio, phần còn lại (dữ liệu chuẩn, Kaggle, Studio, chỉnh actor, nối CARLA) là phần thêm vào.

Tài liệu này nói tool phải làm gì. Cách xây dựng nằm trong `ARCHITECTURE.md`; định dạng dữ liệu và interface nằm trong `CONTRACTS.md`. Ba tài liệu được cập nhật cùng nhau.

Tài liệu mô tả **thiết kế đích**. Repo hiện tại chưa làm được hết; phần chênh lệch và lộ trình refactor nằm ở `ARCHITECTURE.md` mục 7.

Trạng thái: bản nháp, ngày 12/11/2026.

## 1. Mục tiêu

- **Một tool, nhiều approach.** Mỗi approach là một cách dựng cảnh. Mọi approach đọc cùng một bộ dữ liệu chuẩn, cùng một split train/test, đánh giá theo cùng một protocol và ghi kết quả theo cùng một định dạng, nên so sánh được với nhau.
- **Thêm approach không phải sửa lõi.** Approach mới chỉ cần khai báo và thêm phần xử lý còn thiếu.
- **Mỗi approach tự chọn mức hoàn thiện** (mục 4). Có approach chỉ dừng ở mức dựng được cảnh; có approach đi tới mức nối CARLA, chỉnh actor và chạy thử AD stack.

## 2. Người dùng và cách triển khai

- Tool chạy **trên một máy, cho một người dùng**. Không có tài khoản, không có cơ sở dữ liệu dùng chung, không có hàng đợi job nhiều người.
- Máy GPU thuê (RunPod, server của lab) cũng được coi là máy local: cài tool lên đó và mở Studio qua port forward.
- Mentor nhận bàn giao và tự chạy trên máy riêng hoặc máy GPU thuê.
- Train còn chạy được trên **Kaggle** khi máy đang dùng không đủ GPU (mục 7).

## 3. Khái niệm

| Khái niệm | Nghĩa |
|---|---|
| Dữ liệu chuẩn | Một định dạng dữ liệu duy nhất mà mọi approach đọc. Mọi nguồn dữ liệu đều được chuyển về định dạng này trước (mục 6.2). |
| Manifest | File đi kèm dữ liệu chuẩn, ghi cảnh có gì và thiếu gì (LiDAR, box, mô hình camera, mask...). |
| Stage | Một bước xử lý có đầu vào và đầu ra rõ ràng, ví dụ sinh mask, khởi tạo điểm, train, đánh giá, export. |
| Approach | Một cách dựng cảnh: danh sách stage chạy lần lượt, kèm mức năng lực và điều kiện về dữ liệu. |
| Ô chọn | Khâu trong approach có nhiều lựa chọn thay thế được cho nhau, ví dụ khởi tạo (LiDAR hay depth), renderer (3DGS, 3DGUT, 3DGEER). Chọn trước khi chạy. |
| Preset | Tổ hợp ô chọn đặt sẵn, ví dụ `paper`, `no_lidar`. |
| Chuẩn bị riêng | Các stage lấy bớt, sinh thêm hoặc đổi dạng dữ liệu chuẩn cho vừa một approach (mục 6.3). |
| Codebase | Mã nguồn tái dựng mà approach chạy trên đó. Lõi là DriveStudio trong repo này; approach từ codebase khác (NuRec, feed-forward...) được bọc lại và chạy trong môi trường riêng. |
| Track | Loại rig camera: **pinhole** (4–6 camera) hoặc **fisheye** (4–6 camera). Hai track xử lý riêng, không trộn camera pinhole và fisheye trong một lần dựng. |
| Run | Một lần chạy một approach trên một cảnh, với một split và một bộ tham số. |

## 4. Mức năng lực của approach

Mỗi approach khai báo mức cao nhất đạt được. Studio chỉ hiện những tính năng thuộc mức đó trở xuống.

| Mức | Approach phải cung cấp | Tính năng mở ra |
|---|---|---|
| **L1 – Dựng cảnh** (bắt buộc) | Chạy hết; ghi kết quả theo định dạng chuẩn, gồm metric và ảnh render của tập test | Theo dõi tiến trình, xem metric, video theo từng camera, so sánh run |
| **L2 – Render tự do** | Render được ảnh từ một pose camera bất kỳ, với mô hình camera bất kỳ mà approach hỗ trợ | Video theo quỹ đạo mới (dịch làn, đổi độ cao, đổi góc) |
| **L3 – Actor** | Liệt kê actor, đặt lại quỹ đạo actor, ẩn, xoá hoặc thêm actor | Chỉnh actor và quỹ đạo actor trong Studio |
| **L4 – Mô phỏng** | Chạy được sau server gRPC theo giao thức `SensorsimService` của NVIDIA NuRec | Nối CARLA, chạy thử AD stack đơn giản theo vòng kín, chấm điểm |

Nối CARLA, chỉnh actor, chỉnh quỹ đạo actor và chạy thử AD stack là tính năng **tuỳ approach**. Approach chỉ đạt L1 vẫn hợp lệ.

## 5. Luồng sử dụng

1. **Input:** chọn thư mục dữ liệu. Tool nhận dạng loại nguồn (dataset nào, CARLA hay dữ liệu riêng).
2. **Ingest và kiểm tra:** chuyển toàn bộ dữ liệu sang dữ liệu chuẩn, không bỏ bớt thứ gì; kiểm tra đồng bộ thời gian, calib, mô hình ống kính, FOV, ảnh hỏng hoặc thiếu; ghi manifest và báo cáo kiểm tra. Bước này không phụ thuộc approach và chỉ chạy một lần cho mỗi cảnh.
3. **Chọn đoạn và split:** chọn đoạn khung cần dựng, tạo split train/test.
4. **Chọn approach:** chọn approach, rồi chọn preset hoặc chỉnh từng ô chọn. Approach hoặc lựa chọn không chạy được với dữ liệu đang có (ví dụ `init.lidar` khi cảnh không có LiDAR), hoặc tổ hợp không hợp lệ (ví dụ 3DGEER với CamPose bật), bị làm mờ kèm lý do. Màn hình hiện trước danh sách stage sẽ chạy và stage nào dùng lại kết quả cũ.
5. **Chạy và theo dõi:** chọn nơi chạy (mục 7) rồi theo dõi từng stage. Các stage chuẩn bị riêng (mục 6.3) chạy trước; kết quả của chúng được lưu lại để approach khác dùng chung.
6. **Studio:** xem và so sánh kết quả (mục 8).
7. **Xuất và phục vụ:** đóng gói kết quả; với approach đạt L4, bật server gRPC để nối CARLA.

Muốn thử lựa chọn khác thì quay lại bước 4, đổi ô chọn hoặc tham số rồi chạy lại; stage nào không đổi đầu vào thì dùng lại kết quả cũ.

Trên giao diện có thể cho chọn approach sớm để người dùng biết trước, nhưng việc xử lý dữ liệu luôn đi theo thứ tự trên. Mọi bước đều làm được bằng CLI (`d2s <nhóm> <hành động>`, `CONTRACTS.md` mục 10.4), không bắt buộc qua Studio; Studio và CLI dùng chung một bước kiểm tra điều kiện và ràng buộc.

## 6. Dữ liệu

### 6.1 Nguồn dữ liệu được hỗ trợ

- Waymo, nuScenes, nuPlan, Argoverse 2, PandaSet, KITTI, PhysicalAI.
- CARLA: dữ liệu sinh từ CARLA cho track fisheye (rig 4–6 camera fisheye, có pose, depth và box ground truth).
- Dữ liệu riêng của mentor: định dạng chưa biết; đi qua một converter riêng khi có thông tin.
- Ảnh kèm pose dạng COLMAP.

### 6.2 Dữ liệu chuẩn

Mọi nguồn được chuyển sang **một** định dạng chung trước khi tới approach. Định dạng phải có:

- **Rig:** từng camera có id, mô hình ống kính, intrinsics, hệ số méo, độ phân giải, extrinsic so với thân xe, loại shutter.
- **Mô hình ống kính:** `opencv_pinhole`, `opencv_fisheye`, `ftheta`; đúng ba loại trong `CameraSpec` của NuRec.
- **Pose của xe** theo thời gian, đơn vị mét, có ghi rõ hệ toạ độ.
- **Ảnh** của từng camera theo từng khung, kèm thời điểm bắt đầu và kết thúc phơi sáng.
- **Tuỳ chọn:** LiDAR, box 3D theo thời gian, mask.
- **Manifest** ghi những gì có và thiếu; bước 4 của luồng sử dụng dựa vào đây để khoá approach.

Thiết kế chọn **NCore V4** của NVIDIA làm định dạng chuẩn (lý do ở `ARCHITECTURE.md` mục 2.2).

### 6.3 Chuẩn bị riêng cho approach

Mỗi approach khai báo nó cần gì. Từ dữ liệu chuẩn, các stage chuẩn bị:

- **lấy bớt:** chọn camera, cắt đoạn khung, giảm độ phân giải, khử méo về pinhole;
- **sinh thêm** dữ liệu thiếu: mask bầu trời, mask vật động, tư thế người (SMPL), depth prior, box từ detector;
- **đổi dạng** sang thứ codebase của approach đọc được.

Dữ liệu sinh thêm được lưu lại theo cảnh, stage, phiên bản và tham số, dùng chung giữa các approach.

### 6.4 Split

Mỗi cảnh có file split ghi khung và camera nào để train, khung và camera nào để test. Mọi approach đánh giá trên cùng split. Không đánh giá trên khung đã dùng để train. Codebase có thể chỉ hỗ trợ một số kiểu split; khi đó approach bị khoá với split không hỗ trợ.

## 7. Nơi chạy

| Chế độ | Khi nào dùng | Phạm vi |
|---|---|---|
| **Local** | Máy đang chạy tool có GPU đủ mạnh, kể cả máy GPU thuê | Đầy đủ L1–L4 |
| **Kaggle** | Máy đang dùng không đủ GPU | Ingest, chuẩn bị, train và đánh giá (L1) |
| **CLI** | Chạy không qua Studio, kể cả trong notebook viết tay | Như chế độ tương ứng |

Yêu cầu riêng cho chế độ Kaggle:

- Tool tự upload dữ liệu lên Kaggle dưới dạng dataset private, tự sinh notebook, đẩy lên và ra lệnh chạy qua Kaggle API. Người dùng chỉ cần cài API token một lần.
- Studio hiện trạng thái job và log; khi job xong, tool tự tải kết quả về.
- Kết quả được lưu sau từng bước, kể cả checkpoint giữa chừng. Khi một phiên Kaggle hết giờ, lần chạy sau tiếp tục từ chỗ dở.
- Bật Studio ngay trong phiên Kaggle qua tunnel chỉ là tính năng phụ, không bảo đảm hoạt động.

## 8. Studio

Studio là giao diện web chạy trên máy đang dùng. Nhóm A–D chỉ cần approach đạt L1. Trừ viewer 3DGS ở chế độ chính, nhóm A–D không cần GPU CUDA.

**A. Dữ liệu đầu vào**
- Tổng quan cảnh: nguồn, số khung, thời lượng, những gì có và thiếu (từ manifest).
- Sơ đồ rig nhìn từ trên xuống: vị trí, hướng, FOV, loại ống kính của từng camera.
- Quỹ đạo xe trên bản đồ 2D, có thanh thời gian.
- Lưới ảnh đồng bộ của tất cả camera; chồng được box 3D, mask, điểm LiDAR lên ảnh.
- Báo cáo kiểm tra dữ liệu.
- Split train/test hiện trên thanh thời gian.

**B. Tiến trình**
- Các stage của run với trạng thái (chờ, đang chạy, xong, lỗi, dùng lại kết quả cũ); ghi rõ stage nào dùng LiDAR.
- Thời gian, VRAM, log và loss.
- Trạng thái job khi chạy trên Kaggle.

**C. Kết quả của một run**
- Bảng metric: PSNR, SSIM, LPIPS trên tập test, tách theo camera; PSNR trong vùng vật động, vùng người, vùng xe nếu cảnh có box; thời gian train, VRAM, số Gaussian.
- Trình phát video theo camera: ảnh gốc, ảnh render, bản đồ sai khác; thanh trượt so sánh; phát đồng bộ tất cả camera.
- Depth render và tách lớp nền, actor nếu approach xuất ra.
- Viewer 3DGS: đi trong cảnh, xem quỹ đạo xe, vị trí camera, bật tắt box actor. Có hai chế độ: chế độ chính render bằng renderer của approach (cần GPU CUDA, đúng hoàn toàn với model); chế độ xem trước vẽ ngay trong trình duyệt (không cần GPU CUDA, gần đúng, chỉ camera pinhole). Chi tiết ở `ARCHITECTURE.md` mục 2.12, 2.13 và 4.9.

**D. So sánh nhiều run**
- Bảng xếp hạng các approach trên cùng cảnh và cùng split.
- Video của nhiều approach đặt cạnh nhau theo cùng camera, cùng thời điểm.
- Biểu đồ chất lượng theo thời gian train và theo VRAM.
- Xuất bảng và hình cho báo cáo.

**E. Tính năng theo mức của approach** (cần GPU CUDA)
- L2: đặt quỹ đạo camera mới và render video; đổi mô hình camera khi render.
- L3: danh sách actor; ẩn, xoá, thêm actor; sửa quỹ đạo actor trên bản đồ 2D; render lại.
- L4: bật server gRPC; nối CARLA; xem đồng thời camera dựng lại và camera CARLA; chạy AD stack đơn giản theo vòng kín; xem quỹ đạo của agent và điểm số; phát lại một lượt chạy.

Nút của mức mà approach không đạt được ẩn đi, kèm ghi chú.

## 9. Đánh giá

- Cùng split, cùng độ phân giải, đo trên ảnh 8 bit.
- PSNR, SSIM, LPIPS; ghi rõ backbone và phiên bản của LPIPS.
- PSNR trong vùng vật động, vùng người, vùng xe khi cảnh có box.
- Thời gian train, VRAM cao nhất, số Gaussian.
- Với mỗi run, ghi rõ LiDAR được dùng ở stage nào, hoặc không dùng.
- Ghi rõ các ô chọn đã dùng, để so sánh theo từng khâu.

## 10. Kết nối mô phỏng

- Cảnh dựng lại nối với CARLA qua server gRPC theo giao thức `SensorsimService` của NuRec (Apache-2.0), như cách NuRec nối với CARLA. CARLA giữ map, xe ego, actor và vật lý; server của tool chỉ render ảnh cho các camera dựng lại.
- Không cần trích mesh va chạm.
- Khi xe ego đi lệch xa quỹ đạo gốc, ảnh render có thể không khớp với thế giới vật lý của CARLA. Điểm này phải ghi vào danh sách thành phần còn thiếu so với một mô phỏng đầy đủ.

## 11. Yêu cầu chung

- Nhẹ: một tiến trình phục vụ cả API lẫn giao diện; trạng thái lưu trong thư mục, không cần cơ sở dữ liệu.
- Approach từ codebase khác chạy trong môi trường riêng; các approach dòng DriveStudio dùng chung môi trường của repo.
- Mọi run ghi kết quả theo cùng một định dạng, kèm cấu hình đầy đủ, phiên bản mã nguồn và thông tin phần cứng để chạy lại được.
- Thêm approach mới chỉ cần khai báo và thêm stage còn thiếu; không sửa engine, Studio hay server.
- Tên file, định danh, tham số và lệnh CLI theo một bộ quy tắc chung (`CONTRACTS.md` mục 10); code mới đặt đúng chỗ theo `ARCHITECTURE.md` mục 3.1.
- Toàn bộ mã, kể cả mã tái dựng lấy từ DriveStudio, nằm trong một package `drive2sim` (`ARCHITECTURE.md` mục 3). Mỗi thay đổi hành vi ở phần tái dựng phải qua kiểm tra hồi quy trên cùng cảnh.

## 12. Ngoài phạm vi

- Nhiều người dùng, tài khoản, phân quyền, lưu trữ đám mây dùng chung.
- Trộn camera pinhole và fisheye trong cùng một lần dựng.
- Trích mesh va chạm.

## 13. Nghiệm thu

**Nghiệm thu tầng dữ liệu (sau refactor):**
1. nuScenes mini đi qua ingest sang dữ liệu chuẩn, có manifest và báo cáo kiểm tra.
2. Approach `omnire` (preset `paper`) train trên cùng một cảnh bằng đường dữ liệu chuẩn và bằng loader cũ của DriveStudio cho PSNR test chênh không quá một ngưỡng chốt trước.

**Nghiệm thu bộ khung:**
1. Một approach chạy trọn luồng mục 5 ở chế độ Local và chế độ Kaggle; khi phiên Kaggle bị ngắt, lần sau chạy tiếp từ chỗ dở.
2. Kết quả ghi đúng định dạng chuẩn.
3. Studio hiện đủ nhóm A–D và ẩn nhóm E với approach chỉ đạt L1.
4. Thêm một approach thứ hai, và thêm một lựa chọn mới cho một ô, mà không sửa mã của engine, Studio hay server.

## 14. Câu hỏi mở

1. Định dạng bộ dữ liệu riêng của mentor.
2. AD stack dùng để chạy thử ở mức L4 (agent có sẵn của CARLA hay planner tự viết) và tiêu chí chấm điểm (va chạm, lệch làn, hoàn thành tuyến hay tiêu chí khác).
3. Ngưỡng chênh PSNR chấp nhận được ở nghiệm thu tầng dữ liệu.
