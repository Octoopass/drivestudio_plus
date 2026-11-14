from drive2sim.core.config import load_config
from drive2sim.paths import CONFIGS_DIR
from omegaconf import OmegaConf
import json
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse

import torch
from drive2sim.recon.tools.eval import do_evaluation
from drive2sim.recon.utils.misc import import_str
from drive2sim.recon.utils.backup import backup_project
from drive2sim.recon.utils.logging import MetricLogger, setup_logging
from drive2sim.recon.rendering.videos import render_images, save_videos
from drive2sim.recon.data.driving_dataset import DrivingDataset

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup(args):
    # get config
    # drive2sim: config may inherit another file with `extends` (docs/drive2sim/CONTRACTS.md 10.5)
    cfg = OmegaConf.create(load_config(args.config_file))
    
    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
        
    assert "dataset" in cfg or "data" in cfg, \
        "Please specify dataset in config or data in config"
        
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.create(load_config(
            os.path.join(CONFIGS_DIR, "datasets", f"{dataset_type}.yaml")
        ))
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    
    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    
    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    
    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
        
    # also save a backup copy
    saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")
    
    # Backup codes
    backup_project(
        os.path.join(log_dir, 'backup'), "./", 
        ["drive2sim"], 
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    )
    return cfg

# full checkpoint (model + optimizer state) written at every save, used to resume a run
RESUME_CHECKPOINT = "checkpoint_resume.pth"
# drive2sim: training statistics read by eval.standard (CONTRACTS 6.1). Written together with every
# checkpoint so that, after a resume, the counted time matches the steps that were kept.
TRAIN_STATS = "train_stats.json"


def load_train_stats(log_dir: str) -> dict:
    """Stats saved with the checkpoint we resume from; empty for a fresh run."""
    path = os.path.join(log_dir, TRAIN_STATS)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_train_stats(log_dir: str, trainer, step: int, train_time_s: float, peak_vram_mb: int) -> None:
    """Write train time (training steps only), peak VRAM, Gaussian count and iterations."""
    per_node = {k: int(v) for k, v in trainer.get_gaussian_count().items()}
    stats = {
        "step": int(step),
        "num_iters": int(trainer.num_iters),
        "train_time_s": round(train_time_s, 3),
        "peak_vram_mb": int(peak_vram_mb),
        "num_gaussians": sum(per_node.values()),
        "gaussians_per_node": per_node,
    }
    tmp = os.path.join(log_dir, TRAIN_STATS + ".tmp")
    with open(tmp, "w") as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp, os.path.join(log_dir, TRAIN_STATS))

CAMERAS_FILE = "cameras.json"


def save_cameras(log_dir: str, dataset) -> None:
    """drive2sim: write every camera's per-frame pose and intrinsics in the training world frame.

    Same world frame as the checkpoint, scene.ply and the preview package, so a viewer can open
    a reconstruction at exactly the viewpoint of the rendered videos. Poses are camera-to-world in
    the OpenCV convention (x right, y down, z forward), 16 numbers row-major; intrinsics are for an
    image of width x height pixels (the loaded size).
    """
    cameras = []
    for cam in dataset.pixel_source.camera_data.values():
        c2w = cam.cam_to_worlds.detach().cpu().reshape(-1, 16).tolist()
        K = cam.intrinsics.detach().cpu().reshape(-1, 9).tolist()
        cameras.append({
            "name": str(cam.cam_name),
            "width": int(cam.WIDTH),
            "height": int(cam.HEIGHT),
            "c2w": [[round(v, 6) for v in row] for row in c2w],
            "K": [[round(v, 4) for v in row] for row in K],
        })
    payload = {"frame": "drivestudio_world", "convention": "opencv", "units": "metres", "cameras": cameras}
    with open(os.path.join(log_dir, CAMERAS_FILE), "w") as f:
        json.dump(payload, f)


def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)
    save_cameras(cfg.log_dir, dataset)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )
    # training views for FastGS multi-view scoring (only used when trainer.fastgs.enabled)
    trainer.fastgs.view_source = dataset.train_image_set
    
    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        # a full checkpoint (checkpoint_resume.pth) also restores the optimizer state;
        # a model-only checkpoint falls back to a fresh optimizer
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=False
        )
        logger.info(
            f"Resuming training from {args.resume_from}, saved at step {trainer.step}, continuing at step {trainer.step + 1}"
        )
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "Dynamic_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    # setup optimizer  
    trainer.initialize_optimizer()
    
    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    # a checkpoint is written after its step has finished, so a resumed run starts one step later
    start_step = trainer.step + 1 if args.resume_from is not None else trainer.step
    all_iters = np.arange(start_step, trainer.num_iters + 1)
    # drive2sim: resuming after the last step (training finished but the final evaluation was
    # interrupted) leaves all_iters empty; keep `step` defined so the evaluation below still runs.
    step = start_step - 1

    # drive2sim: count only the time spent in training steps (not visualisation, error caching,
    # checkpoint saving, start-up or the final evaluation); a resumed run continues from the time
    # stored with its checkpoint.
    previous = load_train_stats(cfg.log_dir) if args.resume_from is not None else {}
    train_time_s = float(previous.get("train_time_s", 0.0))
    peak_vram_mb = int(previous.get("peak_vram_mb", 0))
    torch.cuda.reset_peak_memory_stats()
    
    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        #----------------------------------------------------------------------------
        #----------------------------     Validate     ------------------------------
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            logger.info("Visualizing...")
            vis_timestep = np.linspace(
                0,
                dataset.num_img_timesteps,
                trainer.num_iters // cfg.logging.vis_freq + 1,
                endpoint=False,
                dtype=int,
            )[step // cfg.logging.vis_freq]
            with torch.no_grad():
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                    compute_metrics=True,
                    compute_error_map=cfg.render.vis_error,
                    vis_indices=[
                        vis_timestep * dataset.pixel_source.num_cams + i
                        for i in range(dataset.pixel_source.num_cams)
                    ],
                )
            if args.enable_wandb:
                wandb.log(
                    {
                        "image_metrics/psnr": render_results["psnr"],
                        "image_metrics/ssim": render_results["ssim"],
                        "image_metrics/occupied_psnr": render_results["occupied_psnr"],
                        "image_metrics/occupied_ssim": render_results["occupied_ssim"],
                    }
                )
            vis_frame_dict = save_videos(
                render_results,
                save_pth=os.path.join(
                    cfg.log_dir, "images", f"step_{step}.png"
                ),  # don't save the video
                layout=dataset.layout,
                num_timestamps=1,
                keys=render_keys,
                save_seperate_video=cfg.logging.save_seperate_video,
                num_cams=dataset.pixel_source.num_cams,
                fps=cfg.render.fps,
                verbose=False,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"image_rendering/" + k: wandb.Image(v)})
            del render_results
            torch.cuda.empty_cache()
                
        
        #----------------------------------------------------------------------------
        #----------------------------  training step  -------------------------------
        # prepare for training
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad() # zero grad
        
        # get data
        train_step_camera_downscale = trainer._get_downscale_factor()
        image_infos, cam_infos = dataset.train_image_set.next(train_step_camera_downscale)
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)
        
        # forward & backward
        outputs = trainer(image_infos, cam_infos)
        trainer.update_visibility_filter()

        loss_dict = trainer.compute_losses(
            outputs=outputs,
            image_infos=image_infos,
            cam_infos=cam_infos,
        )
        # check nan or inf
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in loss {k} at step {step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in loss {k} at step {step}")
        trainer.backward(loss_dict)
        
        # after training step
        trainer.postprocess_per_train_step(step=step)
        torch.cuda.synchronize()
        train_time_s += time.perf_counter() - step_started
        
        #----------------------------------------------------------------------------
        #-------------------------------  logging  ----------------------------------
        with torch.no_grad():
            # cal stats
            metric_dict = trainer.compute_metrics(
                outputs=outputs,
                image_infos=image_infos,
            )
        metric_logger.update(**{"train_metrics/"+k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/"+k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_" + group['name']: group['lr'] for group in trainer.optimizer.param_groups})
        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        #----------------------------------------------------------------------------
        #----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
            (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        )
        if do_save:  
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() // (1024 * 1024))
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=step == trainer.num_iters,
            )
            # drive2sim: full checkpoint (model + optimizer), overwritten at every save, for
            # --resume_from. Also written at the last step and only removed after the final
            # evaluation: if the session ends during evaluation, the rerun resumes past the last
            # step and only evaluates, instead of training again from step 0.
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=False,
                name=RESUME_CHECKPOINT,
            )
            # Stats after the checkpoints: a cut between the two can only leave the stats one
            # save behind (that interval's time is missing), never ahead of the checkpoint (the
            # interval would be counted twice on resume).
            save_train_stats(cfg.log_dir, trainer, step, train_time_s, peak_vram_mb)
        
        #----------------------------------------------------------------------------
        #------------------------    Cache Image Error    ---------------------------
        if (
            step > 0 and trainer.optim_general.cache_buffer_freq > 0
            and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            logger.info("Caching image error...")
            trainer.set_eval()
            with torch.no_grad():
                dataset.pixel_source.update_downscale_factor(
                    1 / dataset.pixel_source.buffer_downscale
                )
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                )
                dataset.pixel_source.reset_downscale_factor()
                dataset.pixel_source.update_image_error_maps(render_results)

                # save error maps
                merged_error_video = dataset.pixel_source.get_image_error_video(
                    dataset.layout
                )
                imageio.mimsave(
                    os.path.join(
                        cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                    ),
                    merged_error_video,
                    fps=cfg.render.fps,
                )
            logger.info("Done caching rgb error maps")
            
    
    logger.info("Training done!")

    do_evaluation(
        step=step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
    )
    # drive2sim: training and evaluation are done; checkpoint_final.pth is kept.
    resume_ckpt = os.path.join(cfg.log_dir, RESUME_CHECKPOINT)
    if os.path.exists(resume_ckpt):
        os.remove(resume_ckpt)
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
    
    return step

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)
    
    # eval
    parser.add_argument("--resume_from", default=None, help=f"checkpoint to resume from; use <log_dir>/{RESUME_CHECKPOINT} to also restore the optimizer state", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    
    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str, help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="omnire", type=str, help="wandb run name, also used to enhance log_dir")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    final_step = main(args)
