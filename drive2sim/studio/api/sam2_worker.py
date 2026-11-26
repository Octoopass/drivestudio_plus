"""Tiến trình SAM2.1 cho công cụ vẽ mask thân xe của Studio (``ego_mask_tool.py``).

Chạy bằng Python có ``transformers>=4.56`` (có ``Sam2Model``), tách khỏi môi trường ``drive2sim``
(transformers cũ hơn, DriveStudio cần): ``ego_mask_tool`` khởi động file này một lần và giữ model
trong bộ nhớ. Giao tiếp qua stdin/stdout, mỗi dòng một JSON:

- vào: ``{"image": <ảnh>, "points": [[x, y, nhãn], ...], "out": <png>}``; ``x``, ``y`` là pixel của
  ảnh gốc, nhãn 1 là thân xe, 0 là không phải;
- ra: ``{"ok": true, "coverage": <tỉ lệ pixel thuộc mask>}`` hoặc ``{"ok": false, "error": ...}``.

Mask (trắng: thân xe) ghi ra ``out`` đúng kích thước ảnh. SAM2 trả ba mask ứng viên khi chỉ có một
điểm; lấy mask nhỏ nhất trong các mask có điểm tin cậy từ 90% mask tốt nhất. Phần nhúng ảnh được
nhớ theo đường dẫn ảnh, nên các lần bấm sau trên cùng ảnh chỉ chạy phần giải mã (nhanh).

Chạy: ``python sam2_worker.py <thư mục model>``; dòng đầu ra ``{"ready": true}`` khi đã nạp xong.
"""

import json
import sys


def main() -> None:
    """Nạp model rồi xử lý từng yêu cầu trên stdin cho tới khi stdin đóng."""
    import cv2
    import numpy as np
    import torch
    from transformers import Sam2Model, Sam2Processor

    model_dir = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam2Processor.from_pretrained(model_dir)
    model = Sam2Model.from_pretrained(model_dir).to(device).eval()
    cache: dict[str, tuple] = {}
    print(json.dumps({"ready": True, "device": device}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            image_path = request["image"]
            if image_path not in cache:
                image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    embeddings = model.get_image_embeddings(inputs["pixel_values"])
                cache.clear()  # một ảnh mỗi lần: đủ cho công cụ, ít bộ nhớ GPU
                cache[image_path] = (image, embeddings, inputs["original_sizes"])
            image, embeddings, sizes = cache[image_path]
            points = request["points"]
            inputs = processor(
                images=image,
                input_points=[[[[float(x), float(y)] for x, y, _ in points]]],
                input_labels=[[[int(label) for _, _, label in points]]],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output = model(
                    input_points=inputs["input_points"],
                    input_labels=inputs["input_labels"],
                    image_embeddings=embeddings,
                    multimask_output=len(points) == 1,
                )
            masks = processor.post_process_masks(output.pred_masks.cpu(), sizes.cpu())[0][0]
            scores = output.iou_scores[0, 0]
            # Một điểm: ba ứng viên lồng nhau (phần, vật, cả vùng). Thân xe thường là mảng nhỏ ở
            # mép ảnh, nên lấy ứng viên nhỏ nhất trong các ứng viên gần tốt như ứng viên tốt nhất.
            good = [i for i in range(len(scores)) if scores[i] >= 0.9 * scores.max()]
            best = min(good, key=lambda i: int(masks[i].sum()))
            mask = masks[best].numpy().astype(bool)
            cv2.imwrite(request["out"], mask.astype(np.uint8) * 255)
            print(json.dumps({"ok": True, "coverage": float(mask.mean())}), flush=True)
        except Exception as error:  # noqa: BLE001 - báo lỗi cho Studio, tiến trình chạy tiếp
            message = f"{type(error).__name__}: {error}"
            print(json.dumps({"ok": False, "error": message}), flush=True)


if __name__ == "__main__":
    main()
