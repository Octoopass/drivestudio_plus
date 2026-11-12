"""
@file   extract_masks_hf.py
@brief  Drop-in replacement for extract_masks.py using the Hugging Face port of
        SegFormer-B5 (Cityscapes, 1024x1024). Runs in the main drivestudio env,
        so no separate torch 1.8 / mmcv-full 1.2.7 environment is needed.

Requires: pip install transformers

Usage (same outputs and flags as extract_masks.py, minus the SegFormer paths):
    python -m drive2sim.recon.data.tools.extract_masks_hf \
        --data_root data/nuscenes/processed_10Hz/mini \
        --start_idx 0 --num_scenes 10 --process_dynamic_mask
"""
import os
from argparse import ArgumentParser
from glob import glob

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

# Cityscapes train ids: 10 = sky, 11/12/17/18 = person/rider/motorcycle/bicycle, 13/14/15 = car/truck/bus
dataset_classes_in_sematic = {
    'Vehicle': [13, 14, 15],
    'human': [11, 12, 17, 18],
}
# SegFormer-B5 Cityscapes config resizes to fit within (2048, 1024), keeping aspect ratio
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def segment(model, img, device):
    """img: HxWx3 uint8 -> HxW uint8 Cityscapes train-id map."""
    h, w = img.shape[:2]
    # never upscale: fp32 at 1821x1024 (the mmseg upscale of 1600x900) does not fit in 4 GB
    scale = min(1., 2048 / max(h, w), 1024 / min(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    x = torch.from_numpy(img).to(device).permute(2, 0, 1)[None].float() / 255.
    x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    logits = model(pixel_values=x).logits  # fp16/autocast give NaN logits with this checkpoint
    logits = F.interpolate(logits.float(), size=(h, w), mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].byte().cpu().numpy()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument("--scene_ids", default=None, type=int, nargs="+",
                        help="scene ids to be processed, a list of integers separated by space")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="If no scene id or split_file is given, use start_idx and num_scenes")
    parser.add_argument("--num_scenes", type=int, default=200)
    parser.add_argument('--process_dynamic_mask', action='store_true',
                        help="Whether to process fine dynamic masks")
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--model', type=str, default='nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        split_file = open(args.split_file, "r").readlines()[1:]
        if "kitti" in args.split_file or "nuplan" in args.split_file:
            scene_ids_list = [line.strip().split(",")[0] for line in split_file]
        else:
            scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(args.device).eval()

    for scene_id in tqdm(scene_ids_list, 'Extracting Masks ...'):
        scene_id = str(scene_id).zfill(3)
        scene_dir = os.path.join(args.data_root, scene_id)
        img_dir = os.path.join(scene_dir, args.rgb_dirname)

        sky_mask_dir = os.path.join(scene_dir, "sky_masks")
        os.makedirs(sky_mask_dir, exist_ok=True)
        if args.process_dynamic_mask:
            rough_human_mask_dir = os.path.join(scene_dir, "dynamic_masks", "human")
            rough_vehicle_mask_dir = os.path.join(scene_dir, "dynamic_masks", "vehicle")
            all_mask_dir = os.path.join(scene_dir, "fine_dynamic_masks", "all")
            human_mask_dir = os.path.join(scene_dir, "fine_dynamic_masks", "human")
            vehicle_mask_dir = os.path.join(scene_dir, "fine_dynamic_masks", "vehicle")
            for d in (all_mask_dir, human_mask_dir, vehicle_mask_dir):
                os.makedirs(d, exist_ok=True)

        for fpath in tqdm(sorted(glob(os.path.join(img_dir, '*'))), f'scene[{scene_id}]'):
            fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]
            if args.ignore_existing and os.path.exists(os.path.join(sky_mask_dir, f"{fbase}.png")):
                continue

            mask = segment(model, imageio.imread(fpath)[..., :3], args.device)

            sky_mask = np.isin(mask, [10])
            imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

            if args.process_dynamic_mask:
                rough_human = imageio.imread(os.path.join(rough_human_mask_dir, f"{fbase}.png")) > 0
                valid_human = np.logical_and(np.isin(mask, dataset_classes_in_sematic['human']), rough_human)
                imageio.imwrite(os.path.join(human_mask_dir, f"{fbase}.png"), valid_human.astype(np.uint8) * 255)

                rough_vehicle = imageio.imread(os.path.join(rough_vehicle_mask_dir, f"{fbase}.png")) > 0
                valid_vehicle = np.logical_and(np.isin(mask, dataset_classes_in_sematic['Vehicle']), rough_vehicle)
                imageio.imwrite(os.path.join(vehicle_mask_dir, f"{fbase}.png"), valid_vehicle.astype(np.uint8) * 255)

                valid_all = np.logical_or(valid_human, valid_vehicle)
                imageio.imwrite(os.path.join(all_mask_dir, f"{fbase}.png"), valid_all.astype(np.uint8) * 255)
