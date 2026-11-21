| Thuật ngữ | Nghĩa trong báo cáo | Ví dụ |
|---|---|---|
| Method (approach) | Một cách dựng cảnh có code riêng. Trong tool drive2sim, mỗi method là một approach. | OmniRe, Street GS, PVG, DeformableGS, Instant NuRec, MCGS-SLAM |
| Cấu hình | Một method cộng các lựa chọn bên trong nó: renderer, luật densify, kỹ thuật tăng tốc. Mỗi run chạy đúng một cấu hình. | OmniRe + 3DGUT + FastGS + Faster-GS |
| Renderer | Cách vẽ Gaussian ra ảnh. 3DGS chỉ vẽ được camera pinhole; 3DGUT và 3DGEER vẽ được cả camera fisheye. | 3DGS, 3DGUT, 3DGEER |
| Luật densify | Cách thêm và bớt Gaussian trong lúc train. | gốc, MCMC, FastGS |
| Dựng từng cảnh (per-scene optimization) | Tối ưu một mô hình riêng cho mỗi cảnh, thường mất vài giờ GPU. | OmniRe, Street GS |
| Dựng tức thì (feed-forward) | Mạng đã train sẵn dựng cảnh mới trong vài giây, không tối ưu lại cho từng cảnh. | Instant NuRec |
| Track benchmark A, B, C | Ba bảng xếp hạng riêng của benchmark. A: dựng từng cảnh trên đoạn ngắn (khoảng 200 khung). B: quãng đường dài từ 1 km. C: dựng tức thì (feed-forward). | Track A có ba ô A-LB, A-L, A-0 |
| Ô A-LB, A-L, A-0 | Ba ô của track A, chia theo đầu vào: có LiDAR và box 3D (A-LB), chỉ có LiDAR (A-L), không có cả hai (A-0). | |
| Vùng che fisheye 88° | Khi train và đánh giá, che các pixel có tia sáng lệch khỏi trục quang của camera quá 88°. Góc nhìn còn lại là 2 × 88° = 176°, không phải FOV 88°. | KITTI-360: ống kính nhìn tới 96,1° và 92,5° lệch trục |
| Pinhole, fisheye | Loại camera: pinhole là camera thường, fisheye là camera góc rất rộng (khoảng 180°). Báo cáo không gọi loại camera là "track". | KITTI-360: 2 pinhole, 2 fisheye |
| ID theo dõi (track ID) | Mã gắn cho cùng một vật thể qua các khung, để biết box ở hai khung là cùng một xe. | |
| Mốc M0–M5 | Các mốc của kế hoạch tổng thể dự án (mục 2). | M2: benchmark nhiều method |
