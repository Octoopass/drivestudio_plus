import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .backends import BACKENDS
from .demo import make_demo
from .fusion import fuse_images
from .jobs import make_bundle
from .manifest import load_manifest
from .models import ProjectManifest
from .sample import sample_dataset, sample_project

WORKSPACE = Path(os.environ.get("FUSION_WORKSPACE", Path(__file__).resolve().parent.parent)).resolve()
STATIC = Path(__file__).parent / "static"


class Job(BaseModel):
    manifest: str
    camera_ids: list[str] | None = None
    frame: int = Field(default=0, ge=0)
    width: Literal[512, 1024, 2048] = 1024

class SampleJob(Job):
    output: str
    max_frames: int = Field(default=4, ge=1, le=32)
    max_mib: int = Field(default=50, ge=1, le=200)
    include_lidar: bool = False
    write: bool = False

class KittiJob(BaseModel):
    dataset_root: str
    sequence: str
    frame_ids: list[int] = Field(min_length=1, max_length=32)
    camera_ids: list[str] = Field(default_factory=lambda: ["image_02", "image_03"])
    output: str
    max_mib: int = Field(default=50, ge=1, le=200)
    write: bool = False

class ReconstructionJob(BaseModel):
    repo: str
    python: str
    checkpoint: str
    ncore_path: str
    model: Literal["pa-front", "pa-multiview", "pq-front"] = "pa-front"
    max_chunks: int = Field(default=1, ge=1, le=8)
    n_gaussians: int = Field(default=200000, ge=100000, le=2000000)
    render_preview: bool = False
    include_dynamic: bool = False
    camera_ids: list[str] = Field(default_factory=list, max_length=5)
    gpu: int | None = Field(default=None, ge=0, le=63)

class ConvertJob(BaseModel):
    dataset: str = Field(min_length=1, max_length=512)
    sequence: str = Field(min_length=1, max_length=80)
    start: int = Field(default=0, ge=0)
    frames: int = Field(default=40, ge=2, le=2000)
    cameras: list[str] = Field(default_factory=lambda: ["image_00", "image_01", "image_02", "image_03"], max_length=4)
    lidar: bool = True
    name: str = ""
    fisheye_model: Literal["rectify120", "ftheta", "opencv"] = "rectify120"

class EvaluateJob(BaseModel):
    ply: str = Field(min_length=1, max_length=512)
    clip: str = Field(min_length=1, max_length=512)
    every: int = Field(default=4, ge=1, le=100)
    gpu: int | None = Field(default=None, ge=0, le=63)

class DriveStudioJob(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    method: str = Field(min_length=1, max_length=200)
    dataset: str = Field(min_length=1, max_length=200)
    render_mode: Literal["", "default", "ut", "geer"] = ""
    fastgs: bool | None = None
    data_root: str = Field(default="", max_length=200)
    steps: int = Field(default=30000, ge=100, le=100000)
    gpu: int | None = Field(default=None, ge=0, le=63)

class PrepareJob(BaseModel):
    dataset: str = Field(min_length=1, max_length=512)
    sequence: str = Field(min_length=1, max_length=80)
    start: int = Field(default=0, ge=0)
    frames: int = Field(default=100, ge=2, le=2000)
    name: str = ""
    objects: bool = True

class ViewerJob(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    gpu: int | None = Field(default=None, ge=0, le=63)

class VideoCompareJob(BaseModel):
    video_a: str = Field(min_length=4, max_length=300)
    video_b: str = Field(min_length=4, max_length=300)
    layout: Literal["vstack", "hstack"] = "vstack"

class DriveSyncJob(BaseModel):
    mode: Literal["dataset", "runs", "folder"] = "dataset"
    dataset: str = Field(default="", pattern=r"^[a-z0-9]{0,32}$")
    items: list[str] = Field(default_factory=list, max_length=200)
    drive_path: str = Field(default="", max_length=512)
    sequence: str = Field(default="", max_length=80)
    session: str = Field(default="", pattern=r"^[A-Za-z0-9_.-]{0,64}$")       # empty = settings colab.data_session
    dest: str = "" 

class ColabSession(BaseModel):
    session: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")

class ColabNew(ColabSession):
    gpu: str = Field(default="", max_length=16)          # one of settings colab.gpus, or "" for CPU

class ColabTrainJob(DriveStudioJob):
    session: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    colab_gpu: str = Field(default="", max_length=16)    # one of settings colab.gpus; "" = default_gpu
    scene: int = Field(default=0, ge=0, le=999)
    with_ckpt: bool = True

class NurecSetup(BaseModel):
    models: list[Literal["pa-multiview", "pa-front", "pq-front"]] = Field(
        default=["pa-multiview", "pa-front", "pq-front"], min_length=1, max_length=3)

class ColabAccount(BaseModel):
    email: str = Field(min_length=3, max_length=254)

class ColabCode(BaseModel):
    code: str = Field(min_length=10, max_length=300)

class UploadItem(BaseModel):
    name: str = Field(min_length=1, max_length=512)
    path: str = Field(default="", max_length=1024)
    size: int = Field(ge=1)

class UploadPlan(BaseModel):
    files: list[UploadItem] = Field(min_length=1, max_length=20000)
    dataset: str = Field(default="auto", pattern=r"^[a-z0-9]{1,32}$")

class PreprocessJob(BaseModel):
    dataset: str = Field(pattern=r"^[a-z0-9]{1,32}$")
    split: str = Field(default="", pattern=r"^[A-Za-z0-9_.-]{0,40}$")
    scene_ids: str = Field(default="", pattern=r"^[0-9 ,]{0,400}$")
    split_file: str = Field(default="", max_length=200)
    start_idx: int = Field(default=0, ge=0, le=100000)
    num_scenes: int = Field(default=0, ge=0, le=2000)
    workers: int = Field(default=8, ge=1, le=64)
    sky_masks: bool = True

class UploadTarget(BaseModel):
    target: str = Field(min_length=1, max_length=200)

class TrainingInput(BaseModel):
    dataset: str = Field(min_length=1, max_length=512)
    camera_ids: list[str] | None = Field(default=None, max_length=4)

class TrainingJob(TrainingInput):
    steps: int = Field(default=1200, ge=1, le=10000, strict=True)


def png(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(encoded).decode("ascii")


def project(job):
    path = (WORKSPACE / job.manifest).resolve()
    if not path.is_relative_to(WORKSPACE) or path.suffix.lower() != ".json":
        raise ValueError("Manifest must be a JSON file within FUSION_WORKSPACE")
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Manifest exceeds 4 MiB")
    manifest = load_manifest(path)
    payload = manifest.model_dump()
    if job.camera_ids is not None:
        payload["fusion"]["camera_ids"] = job.camera_ids
    payload["fusion"]["resolution"] = [job.width, job.width // 2]
    manifest = ProjectManifest.model_validate(payload)
    if job.frame >= len(manifest.frames):
        raise ValueError("Frame index out of range")
    return path, manifest


def create_app():
    from .pipeline import PipelineManager
    from .reconstruction import ReconstructionManager
    from .training import TrainingManager
    from .drivestudio import DriveStudioRuns
    from .uploads import UploadManager
    from .settings import Settings
    settings = Settings(WORKSPACE)
    uploads = UploadManager(WORKSPACE, ssh_user=settings.data["app"]["ssh_user"], ssh_host=settings.data["app"]["ssh_host"])
    from .colab import ColabHelper
    drivestudio = DriveStudioRuns(WORKSPACE, venv=settings.path("ds_venv"))
    reconstruction = ReconstructionManager(WORKSPACE)
    training = TrainingManager(WORKSPACE)
    pipeline = PipelineManager(WORKSPACE)
    pipeline.default_session = settings.data["colab"]["data_session"]
    pipeline.discovery.eval_venv = settings.path("eval_venv")
    colab = ColabHelper(lambda: pipeline.discovery.colab_bin, gpus=[g["id"] for g in settings.gpus])
    from .colab_train import ColabTrainer
    colab_train = ColabTrainer(WORKSPACE, colab, drivestudio, settings)
    from .deps import Deps
    deps = Deps(WORKSPACE, pipeline, drivestudio.venv, settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        pipeline.shutdown()
        colab_train.shutdown()
        colab.shutdown()
        drivestudio.shutdown()
        training.shutdown()
        reconstruction.shutdown()

    app = FastAPI(title="Fisheye Fusion Lab", lifespan=lifespan)
    # Local-only by default. FUSION_ALLOWED_HOSTS (comma separated) adds hosts, e.g. for a read-only preview proxy.
    extra_hosts = [h.strip() for h in os.environ.get("FUSION_ALLOWED_HOSTS", "").split(",") if h.strip()]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver", *extra_hosts])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"detail": "Cross-origin requests are disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(OSError)
    async def file_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/health")
    def health():
        return {"ok": True, "workspace": str(WORKSPACE)}

    @app.get("/api/backends")
    def backends():
        return BACKENDS

    @app.get("/api/runtime")
    def runtime():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            gpu = result.stdout.strip() if result.returncode == 0 else "Unavailable"
        except (OSError, subprocess.TimeoutExpired):
            gpu = "Unavailable"
        return {"cpu_preview": True, "gpu": gpu, "cuda_training": "GEER local job runner",
                "cuda_inference": "Instant NuRec runner; separate environment and local assets required",
                "cloud": "CPU bundle + Instant NuRec inference notebook"}

    @app.post("/api/training/inspect")
    def training_inspect(job: TrainingInput):
        return training.inspect(job.dataset, job.camera_ids)

    @app.post("/api/training")
    def training_start(job: TrainingJob):
        return training.start(job.dataset, job.camera_ids, job.steps)

    @app.get("/api/training/{job_id}")
    def training_status(job_id: str):
        return training.status(job_id)

    @app.post("/api/training/{job_id}/cancel")
    def training_cancel(job_id: str):
        return training.cancel(job_id)

    @app.get("/api/training/{job_id}/artifacts/{name}")
    def training_artifact(job_id: str, name: str):
        return FileResponse(training.artifact(job_id, name))

    def experiment_file(name, run):
        runs = {"baseline": "real-gs-run", "geer": "geer-run"}
        if run not in runs:
            raise ValueError("Unknown experiment run")
        allowed = {"metrics.json", "scene.ply", "reference.png", "before.png", "after.png"}
        allowed.update(f"novel-{i:02d}.png" for i in range(9))
        if name not in allowed:
            raise ValueError("Unknown experiment artifact")
        root = (WORKSPACE / "outputs" / runs[run]).resolve()
        path = (root / name).resolve()
        if not root.is_relative_to(WORKSPACE) or path.parent != root or not path.is_file():
            raise ValueError("Experiment has not completed or artifact is missing")
        return path

    @app.get("/api/experiment")
    def experiment(run: str = "baseline"):
        path = experiment_file("metrics.json", run)
        if path.stat().st_size > 1024**2:
            raise ValueError("Experiment metadata too large")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/experiment/artifacts/{name:path}")
    def experiment_artifact(name: str, run: str = "baseline"):
        return FileResponse(experiment_file(name, run))

    @app.post("/api/demo")
    def demo(model: Literal["opencv_fisheye", "pinhole"] = "opencv_fisheye"):
        path = make_demo(WORKSPACE / "examples" / f"synthetic_{model}", model)
        return {"manifest": str(path.relative_to(WORKSPACE))}

    @app.post("/api/inspect")
    def inspect(job: Job):
        path, manifest = project(job)
        images = []
        for camera in manifest.cameras:
            ref = manifest.frames[job.frame].images.get(camera.id)
            if not ref:
                continue
            image = cv2.imread(str(path.parent / ref))
            if image is None:
                raise ValueError(f"Cannot decode {ref}")
            if (image.shape[1], image.shape[0]) != camera.resolution:
                raise ValueError(f"Image/calibration resolution mismatch: {camera.id}")
            image = cv2.resize(image, (320, round(320 * image.shape[0] / image.shape[1])))
            images.append({"id": camera.id, "png": png(image)})
        return {"manifest": manifest.model_dump(), "images": images, "frame_count": len(manifest.frames)}

    @app.post("/api/render")
    def render(job: Job):
        path, manifest = project(job)
        image, coverage = fuse_images(manifest, path.parent, job.frame)
        return {"png": png(image[:, :, ::-1]), "mask": png(coverage.astype("uint8") * 255), "coverage": float(coverage.mean()),
                "operation": "rotation-only panorama", "translation_ignored": any(np.any(np.asarray(c.T_rig_from_camera)[:3, 3]) for c in manifest.cameras)}


    @app.post("/api/bundle/{platform}")
    def bundle(platform: Literal["colab", "kaggle"], job: Job):
        path, manifest = project(job)
        manifest.frames = [manifest.frames[job.frame]]
        content = make_bundle(manifest, path.parent, platform)
        return Response(content, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="fusion-{platform}.zip"'})

    @app.post("/api/sample")
    def sample(job: SampleJob):
        path, manifest = project(job)
        destination = (WORKSPACE / job.output).resolve()
        if not destination.is_relative_to(WORKSPACE) or destination == WORKSPACE:
            raise ValueError("Sample output must be within FUSION_WORKSPACE")
        result = sample_dataset(
            path, destination, start=job.frame, max_frames=job.max_frames,
            max_bytes=job.max_mib * 1024**2, camera_ids=job.camera_ids or manifest.fusion.camera_ids,
            include_lidar=job.include_lidar, write=job.write,
        )
        result["manifest"] = str(Path(result["manifest"]).relative_to(WORKSPACE))
        return result

    @app.post("/api/import/kitti360")
    def import_kitti360(job: KittiJob):
        from .kitti360 import build_kitti360_manifest
        root = (WORKSPACE / job.dataset_root).resolve()
        output = (WORKSPACE / job.output).resolve()
        if not root.is_relative_to(WORKSPACE) or not output.is_relative_to(WORKSPACE) or output == WORKSPACE:
            raise ValueError("Dataset and output must be within FUSION_WORKSPACE; use CLI for external datasets")
        manifest = build_kitti360_manifest(
            root, job.sequence, job.frame_ids, camera_ids=job.camera_ids,
            max_bytes=job.max_mib * 1024**2,
        )
        result = sample_project(
            manifest, root, output, max_frames=len(job.frame_ids),
            max_bytes=job.max_mib * 1024**2, write=job.write,
        )
        result["manifest"] = str(Path(result["manifest"]).relative_to(WORKSPACE))
        return result

    @app.post("/api/reconstruction/preflight")
    def reconstruction_preflight(job: ReconstructionJob):
        return reconstruction.preflight(job.model_dump())

    @app.post("/api/reconstruction")
    def reconstruction_start(job: ReconstructionJob):
        return reconstruction.start(job.model_dump())

    @app.get("/api/reconstruction/{job_id}")
    def reconstruction_status(job_id: str):
        return reconstruction.status(job_id)

    @app.post("/api/reconstruction/{job_id}/cancel")
    def reconstruction_cancel(job_id: str):
        return reconstruction.cancel(job_id)

    @app.get("/api/reconstruction/{job_id}/artifacts/{relative:path}")
    def reconstruction_artifact(job_id: str, relative: str):
        path = reconstruction.artifact(job_id, relative)
        return FileResponse(path)

    @app.get("/api/datasets")
    def dataset_profiles():
        from .dataset_profiles import public
        return {"profiles": public(), "local": pipeline.discovery.local_datasets(),
                "split_files": pipeline.discovery.split_files()}

    @app.post("/api/datasets/preprocess")
    def dataset_preprocess(job: PreprocessJob):
        return pipeline.preprocess(job.model_dump(), drivestudio.venv)

    @app.get("/api/upload")
    def upload_status():
        return uploads.status()

    @app.post("/api/upload/plan")
    def upload_plan(job: UploadPlan):
        return uploads.plan([f.model_dump() for f in job.files], job.dataset)

    @app.put("/api/upload/chunk")
    async def upload_chunk(request: Request, target: str, offset: int, total: int):
        data = await request.body()
        return await run_in_threadpool(uploads.write, target, offset, total, data)

    @app.post("/api/upload/cancel")
    def upload_cancel(job: UploadTarget):
        return uploads.cancel(job.target)

    @app.get("/api/config")
    def public_config():
        return settings.public()

    @app.get("/api/deps")
    def deps_status():
        return deps.status()

    @app.post("/api/deps/{dep_id}/install")
    def deps_install(dep_id: str):
        return deps.install(dep_id)

    @app.get("/api/setup")
    def setup():
        report = pipeline.discovery.report()
        report["nurec"]["setup_job"] = pipeline.active("nurec-setup")
        return report

    @app.get("/api/outputs/{relative:path}")
    def output_file(relative: str):
        return FileResponse(pipeline.discovery.output_file(relative))

    @app.post("/api/pipeline/convert")
    def pipeline_convert(job: ConvertJob):
        cfg = job.model_dump()
        if not cfg["name"]:
            cfg.pop("name")
        return pipeline.convert(cfg)

    @app.post("/api/pipeline/evaluate")
    def pipeline_evaluate(job: EvaluateJob):
        return pipeline.evaluate(job.model_dump())

    @app.get("/api/pipeline/{job_id}")
    def pipeline_status(job_id: str):
        return pipeline.status(job_id)

    @app.post("/api/pipeline/{job_id}/cancel")
    def pipeline_cancel(job_id: str):
        return pipeline.cancel(job_id)

    @app.get("/api/pipeline/{job_id}/artifacts/{relative:path}")
    def pipeline_artifact(job_id: str, relative: str):
        return FileResponse(pipeline.artifact(job_id, relative))

    @app.get("/api/drivestudio")
    def drivestudio_overview():
        return {"options": drivestudio.options(), "runs": drivestudio.runs(), "videos": drivestudio.videos()}

    @app.get("/api/videos")
    def list_videos():
        return {"videos": drivestudio.videos()}

    @app.post("/api/videos/compare")
    def compare_videos(job: VideoCompareJob):
        return drivestudio.compare_videos(job.video_a, job.video_b, job.layout)

    @app.get("/api/colab/cli")
    def colab_cli():
        return colab.cli_info()

    @app.post("/api/colab/cli/update")
    def colab_cli_update():
        return colab.cli_update()

    @app.get("/api/colab/accounts")
    def colab_accounts():
        return colab.accounts()

    @app.post("/api/colab/accounts/use")
    def colab_account_use(job: ColabAccount):
        return colab.use_account(job.email)

    @app.post("/api/colab/accounts/forget")
    def colab_account_forget(job: ColabAccount):
        return colab.forget_account(job.email)

    @app.post("/api/colab/accounts/login")
    def colab_login():
        return colab.login_start()

    @app.post("/api/colab/accounts/code")
    def colab_login_code(job: ColabCode):
        return colab.login_code(job.code)

    @app.post("/api/colab/accounts/cancel")
    def colab_login_cancel():
        return colab.login_cancel()

    @app.post("/api/nurec/setup")
    def nurec_setup(job: NurecSetup):
        return pipeline.nurec_setup(job.models)

    @app.get("/api/colab/usage")
    def colab_usage():
        return colab.usage()

    @app.post("/api/colab/sessions/stop")
    def colab_stop(job: ColabSession):
        return colab.stop(job.session)

    @app.get("/api/colab/train")
    def colab_train_jobs():
        return colab_train.jobs()

    @app.post("/api/colab/train")
    def colab_train_start(job: ColabTrainJob):
        return colab_train.start(job.model_dump())

    @app.get("/api/colab/train/{name}")
    def colab_train_status(name: str):
        return colab_train.status(name)

    @app.post("/api/colab/train/{name}/cancel")
    def colab_train_cancel(name: str):
        return colab_train.cancel(name)

    @app.get("/api/colab/sessions")
    def colab_sessions():
        return colab.sessions()

    @app.post("/api/colab/sessions")
    def colab_new(job: ColabNew):
        return colab.create(job.session, job.gpu or None)

    @app.get("/api/colab/url")
    def colab_url(session: str):
        return colab.url(session)

    @app.get("/api/colab/drive")
    def colab_drive(session: str):
        return colab.drive_status(session)

    @app.post("/api/colab/drive/mount")
    def colab_mount(job: ColabSession):
        return colab.mount(job.session)

    @app.get("/api/colab/drive/mount")
    def colab_mount_status(session: str):
        return colab.mount_status(session)

    @app.post("/api/colab/drive/confirm")
    def colab_confirm(job: ColabSession):
        return colab.confirm_mount(job.session)

    @app.get("/api/colab/browse")
    def colab_browse(session: str, path: str = ""):
        return colab.browse(session, path)

    @app.get("/api/colab/find")
    def colab_find(session: str):
        return colab.find(session)

    @app.post("/api/colab/sync")
    def colab_sync(job: DriveSyncJob):
        cfg = job.model_dump()
        if not cfg["dest"]:
            cfg.pop("dest")
        return pipeline.sync_drive(cfg)

    @app.post("/api/drivestudio/prepare")
    def drivestudio_prepare(job: PrepareJob):
        cfg = job.model_dump()
        if not cfg["name"]:
            cfg.pop("name")
        return pipeline.prepare_drivestudio(cfg, drivestudio.venv)

    @app.post("/api/drivestudio/train")
    def drivestudio_train(job: DriveStudioJob):
        return drivestudio.train(job.model_dump())

    @app.get("/api/drivestudio/viewer")
    def drivestudio_viewer():
        return drivestudio.viewer_status()

    @app.post("/api/drivestudio/viewer")
    def drivestudio_viewer_start(job: ViewerJob):
        return drivestudio.viewer_start(job.name, job.gpu)

    @app.post("/api/drivestudio/viewer/stop")
    def drivestudio_viewer_stop():
        return drivestudio.viewer_stop()

    @app.get("/api/drivestudio/runs/{name}")
    def drivestudio_run(name: str):
        run = drivestudio.run(name)
        if run is None:
            return JSONResponse({"detail": "run not found"}, status_code=404)
        return run

    @app.get("/api/drivestudio/runs/{name}/files/{relative:path}")
    def drivestudio_file(name: str, relative: str):
        return FileResponse(drivestudio.file(name, relative))

    @app.get("/api/notebooks/instant-nurec/{platform}")
    def nurec_notebook(platform: Literal["colab", "kaggle"]):
        from .nurec_notebook import make_nurec_notebook
        return Response(
            make_nurec_notebook(platform), media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f'attachment; filename="instant-nurec-{platform}.ipynb"'},
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/legacy")
    def legacy():
        return FileResponse(STATIC / "legacy.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
