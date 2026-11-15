# FastGS-style multi-view densification and pruning

An adaptation of the densification and pruning rules of [FastGS](https://github.com/fastgs/FastGS) to DriveStudio.
It changes which Gaussians are added and removed; it does not change the renderer, so it works with
`render_mode: default`, `ut` and `geer`.

Enable it under `trainer` in a method config (see `streetgs_fastgs.yaml`, `streetgs_geer_fastgs.yaml`, `omnire_fastgs.yaml`):

```yaml
trainer:
  fastgs:
    enabled: true
```

## Giải thích từ nguyên lý gốc

Nền tảng về densify (vì sao phải thêm, bớt Gaussian trong lúc train, và DriveStudio làm thế nào): `densify_3dgut.md` mục 1.

**Câu hỏi đúng.** Densify gốc thêm đốm ở chỗ "bị kéo mạnh" (gradient theo vị trí lớn). Đó chỉ là dấu hiệu gián tiếp: một đốm có thể bị kéo mạnh vì nằm ở rìa ảnh hay góc nhìn lạ trong khi ảnh chỗ đó đã đúng, và lực kéo đo trên từng ảnh một. Câu cần hỏi thật ra là: *đốm này có đang góp vào chỗ ảnh bị sai, ở nhiều góc nhìn, không?* FastGS hỏi trực tiếp câu đó.

**Cách làm (bản của dự án, `drive2sim/recon/speedups/fastgs.py`).** Trước mỗi lần densify hoặc prune:

1. render `num_views` (10) ảnh train rải đều theo đoạn đường;
2. đánh dấu pixel sai nhiều: chênh lệch màu với ảnh thật (chuẩn hoá) vượt `loss_thresh` (0,1);
3. chấm điểm từng đốm: đốm góp bao nhiêu màu vào các pixel bị đánh dấu (theo trọng số blend), trung bình trên các ảnh thật sự thấy đốm;
4. chỉ densify đốm vừa vượt ngưỡng gradient như cũ, vừa có điểm từ `score_thresh` (1,0) trở lên;
5. từ bước `prune_from` (15 000), bỏ thêm đốm mờ (độ đục dưới 0,1) hoặc điểm sai rất cao.

Ví dụ: densify gốc thêm giấy ở mọi chỗ "bị giằng co"; FastGS lùi ra nhìn bức tranh từ 10 góc, khoanh những chỗ thật sự chép sai, chỉ thêm giấy ở đó.

**Vì sao nhanh hơn.** FastGS không làm mỗi phép tính nhanh hơn; nó làm **ít đốm hơn**, mà gần như mọi phần của một bước train tỉ lệ với số đốm: gom và biến đổi đốm theo node của OmniRe, rasterizer, optimizer. Nên nó cắt cả phần mà Faster-GS không chạm tới (`fastergs.md`).

**Đã đo** (OmniRe, nuScenes `scene-0061`, `front3`, 30 000 bước, T4, mỗi cấu hình một lần chạy):

| | Gốc (`2d33`) | FastGS (`453a`) |
|---|---|---|
| Số Gaussian | 1,52 triệu | 0,55 triệu |
| Thời gian train | 3 giờ 39 phút | 2 giờ 47 phút |
| VRAM đỉnh | 6,6 GB | 4,4 GB |
| PSNR / SSIM / LPIPS | 25,15 / 0,786 / 0,251 | 25,02 / 0,777 / 0,268 |

Cái giá: ít đốm thì có chỗ không đủ chi tiết; LPIPS (nhạy với chi tiết) tăng rõ hơn PSNR giảm. Run gốc bật đo profile còn run FastGS không, nên chênh lệch thời gian có thể bị phóng đại một chút. `score_thresh` chưa chỉnh theo cảnh.

**FastGS và renderer.** FastGS chấm điểm bằng chính hàm vẽ với tham số camera của renderer đang dùng, nên chạy với 3DGS, 3DGEER, 3DGUT. Nó vẫn giữ cửa gradient của DriveStudio, nên với 3DGUT (ngưỡng gradient chưa chỉnh, `densify_3dgut.md`) cửa đó gần như luôn mở và việc lọc dựa hết vào điểm sai.

## Where the code is

`drive2sim/recon/speedups/fastgs.py` scores the Gaussians (class `FastGS`); `fastgs_rules.py` holds the defaults, the schedule and the view sampling, and is tested on CPU in `tests/drive2sim/recon/speedups/`. The trainer calls `FastGS.before_step` at the start of a step and `FastGS.clear_masks` after the models refine; the models apply the masks in `refinement_after()` (`fastgs_filter_densify`, `fastgs_extra_culls` in `models/gaussians/vanilla.py`).

## What it does

At every refinement step that densifies or prunes, the trainer first renders `num_views` training views
spread evenly over the clip. In each view it marks the high-error pixels (min-max normalised L1 error above
`loss_thresh`) and measures how much each Gaussian contributes to them.

- **Densification.** A Gaussian is split or duplicated only if it passes DriveStudio's usual gradient test
  *and* its score exceeds `score_thresh`.
- **Pruning.** From `prune_from` on, at the last refinement before each opacity reset, Gaussians with opacity
  below `prune_min_opacity` or a normalised error score above `prune_score_thresh` are removed, in addition
  to DriveStudio's own culling.

Applied to `Background`, `RigidNodes` and `DeformableNodes`. `SMPLNodes` and PVG are unchanged.

## Differences from the paper

1. **Blend-weighted count.** FastGS counts each high-error pixel a Gaussian touches as 1, using its own
   rasterizer. Here each pixel counts with the Gaussian's blending weight (alpha x transmittance), obtained
   as a gradient through the existing rasterizer. FastGS's threshold of 5 therefore does not carry over.
2. **Averaged over the views that see the Gaussian.** FastGS averages over all sampled views. In a
   forward-driving clip most views do not see a given Gaussian, so that rule would block nearly all
   densification.
3. **No pruning budget during densification**, and none of FastGS's rasterizer speed-ups.
4. **Views are spread over each camera.** The sampled views are split evenly between the cameras of
   the rig and then spread over each camera's frames. Spreading them over the whole image list (as
   the first version did) put all 10 views on one camera whenever the number of training frames was
   a multiple of 10, because DriveStudio lists images frame by frame and camera by camera; Gaussians
   seen only by the other cameras then counted as unseen and could not densify. A one-camera rig
   samples exactly as before.

DriveStudio itself (not FastGS) only densifies when `step % reset_alpha_interval` exceeds the number
of training images, so rigs with many cameras or long clips densify less: 1080 images (6 cameras,
200 frames) leave 64% of each 3000-step cycle, more than 3000 images leave none. This is unchanged.

## Tuning

Every scoring pass logs a line such as

```
FastGS @ step 1200 Background: seen in sampled views 412345/500000, may densify 30112, score p50/p90/p99 0.21/1.80/9.40
```

Set `score_thresh` from these percentiles. If "seen in sampled views" is low, raise `num_views` or set
`densify_unseen: true`. Cost is about two extra renders per sampled view per refinement step.
